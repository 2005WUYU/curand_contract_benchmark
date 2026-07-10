from __future__ import annotations

import torch

from flagrand.rng._mtgp32_data import (
    MTGP32_BLOCK_SIZE,
    MTGP32_MAX_BLOCKS,
    MTGP32_MAX_CHUNKS_PER_LAUNCH,
    MTGP32_SEQUENCE_CHUNK,
    build_initial_state,
    build_param_tensors,
)
from flagrand.rng._mtgp32_kernel import launch_mtgp32_blocks
from flagrand.rng._sequence import clear_chunk_cache
from flagrand.rng._stateful_output import RAW_OUTPUT, StatefulOutput


def ensure_working_state(generator, seed_val: int, device_str: str) -> None:
    ws_seed = getattr(generator, "_ws_seed", None)
    ws_device = getattr(generator, "_ws_device", None)
    ws_blocks = getattr(generator, "_ws_blocks", 0)
    if ws_seed == seed_val and ws_device == device_str and ws_blocks >= MTGP32_MAX_BLOCKS:
        return

    generator._working_state = build_initial_state(seed_val, device_str).clone()
    generator._ws_seed = seed_val
    generator._ws_device = device_str
    generator._ws_blocks = generator._working_state.shape[0]
    generator._ws_next_block_start = 0
    generator._parameter_tensors = build_param_tensors(device_str)
    clear_chunk_cache(generator)


def advance_to_block_start(
    generator,
    block_start_element: int,
    device_str: str,
    num_warps: int,
) -> None:
    next_block_start = int(getattr(generator, "_ws_next_block_start", 0))
    if block_start_element < next_block_start:
        return
    blocks_to_skip = (block_start_element - next_block_start) // MTGP32_BLOCK_SIZE
    if blocks_to_skip <= 0:
        return

    scratch = torch.empty(0, device=torch.device(device_str), dtype=torch.int32)
    while blocks_to_skip:
        block_start = (next_block_start % MTGP32_SEQUENCE_CHUNK) // MTGP32_BLOCK_SIZE
        if block_start == 0 and blocks_to_skip >= MTGP32_MAX_BLOCKS:
            launch_chunks = min(
                blocks_to_skip // MTGP32_MAX_BLOCKS,
                MTGP32_MAX_CHUNKS_PER_LAUNCH,
            )
            block_count = MTGP32_MAX_BLOCKS
        else:
            launch_chunks = 1
            block_count = min(blocks_to_skip, MTGP32_MAX_BLOCKS - block_start)

        generate_blocks_into(
            generator,
            scratch,
            device_str,
            num_warps,
            block_start=block_start,
            block_count=block_count,
            chunks=launch_chunks,
            n_elements=0,
            output=RAW_OUTPUT,
        )
        blocks_to_skip -= launch_chunks * block_count
        next_block_start = int(getattr(generator, "_ws_next_block_start", 0))


def generate_blocks_into(
    generator,
    out: torch.Tensor,
    device_str: str,
    num_warps: int,
    *,
    block_start: int,
    block_count: int,
    chunks: int,
    n_elements: int | None = None,
    output: StatefulOutput = RAW_OUTPUT,
) -> None:
    if chunks <= 0 or block_count <= 0:
        return
    if chunks > 1 and (block_start != 0 or block_count != MTGP32_MAX_BLOCKS):
        raise ValueError("MTGP32 multi-chunk launch requires a full 192-block chunk.")

    pos, sh1, sh2, param, temper = generator._parameter_tensors
    block_end = block_start + block_count
    state = generator._working_state[block_start:block_end]
    start_iter = (
        int(getattr(generator, "_ws_next_block_start", 0)) // MTGP32_SEQUENCE_CHUNK
    ) % 4
    output_elements = out.numel() if n_elements is None else int(n_elements)
    launch_mtgp32_blocks(
        out,
        state,
        pos[block_start:block_end],
        sh1[block_start:block_end],
        sh2[block_start:block_end],
        param[block_start * 16 : block_end * 16],
        temper[block_start * 16 : block_end * 16],
        output_elements=output_elements,
        chunks=chunks,
        start_iter=start_iter,
        block_count=block_count,
        output=output,
        requested_num_warps=num_warps,
    )
    generator._ws_next_block_start = (
        int(getattr(generator, "_ws_next_block_start", 0))
        + chunks * block_count * MTGP32_BLOCK_SIZE
    )
