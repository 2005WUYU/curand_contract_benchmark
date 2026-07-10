from __future__ import annotations

import torch

from flagrand.rng._mt19937_data import (
    MT19937_MAX_CHUNKS_PER_LAUNCH,
    MT19937_N,
    MT19937_NUM_STREAMS,
    MT19937_SEQUENCE_CHUNK,
    build_initial_states,
)
from flagrand.rng._mt19937_kernel import launch_mt19937_blocks
from flagrand.rng._sequence import clear_chunk_cache


def ensure_working_state(generator, seed_val: int, device_str: str) -> None:
    ws_seed = getattr(generator, "_ws_seed", None)
    ws_device = getattr(generator, "_ws_device", None)
    ws_blocks = getattr(generator, "_ws_blocks", 0)
    if ws_seed == seed_val and ws_device == device_str and ws_blocks >= MT19937_NUM_STREAMS:
        return

    initial_state = build_initial_states(seed_val).to(torch.device(device_str))
    generator._working_state = initial_state
    generator._scratch = torch.empty_like(initial_state)
    generator._ws_seed = seed_val
    generator._ws_device = device_str
    generator._ws_blocks = initial_state.shape[0]
    generator._ws_next_block_start = 0
    clear_chunk_cache(generator)


def advance_to_block_start(generator, block_start_element: int, device_str: str, num_warps: int) -> None:
    next_block_start = int(getattr(generator, "_ws_next_block_start", 0))
    if block_start_element < next_block_start:
        return
    blocks_to_skip = (block_start_element - next_block_start) // MT19937_N
    if blocks_to_skip <= 0:
        return

    scratch = torch.empty(0, device=torch.device(device_str), dtype=torch.int32)
    while blocks_to_skip:
        block_start = (next_block_start % MT19937_SEQUENCE_CHUNK) // MT19937_N
        if block_start == 0 and blocks_to_skip >= MT19937_NUM_STREAMS:
            launch_chunks = min(blocks_to_skip // MT19937_NUM_STREAMS, MT19937_MAX_CHUNKS_PER_LAUNCH)
            generate_blocks_into(
                generator,
                scratch,
                device_str,
                num_warps,
                block_start=0,
                block_count=MT19937_NUM_STREAMS,
                rounds=launch_chunks,
                n_elements=0,
                output_mode=0,
            )
            skipped = launch_chunks * MT19937_NUM_STREAMS
        else:
            block_count = min(blocks_to_skip, MT19937_NUM_STREAMS - block_start)
            generate_blocks_into(
                generator,
                scratch,
                device_str,
                num_warps,
                block_start=block_start,
                block_count=block_count,
                rounds=1,
                n_elements=0,
                output_mode=0,
            )
            skipped = block_count
        blocks_to_skip -= skipped
        next_block_start = int(getattr(generator, "_ws_next_block_start", 0))


def generate_blocks_into(
    generator,
    out: torch.Tensor,
    device_str: str,
    num_warps: int,
    *,
    block_start: int,
    block_count: int,
    rounds: int,
    n_elements: int | None = None,
    output_mode: int = 0,
) -> None:
    if rounds <= 0 or block_count <= 0:
        return
    if rounds > 1 and (block_start != 0 or block_count != MT19937_NUM_STREAMS):
        raise ValueError("MT19937 multi-round launch requires a full stream chunk.")

    block_end = block_start + block_count
    state = generator._working_state[block_start:block_end]
    scratch = generator._scratch[block_start:block_end]
    output_elements = out.numel() if n_elements is None else int(n_elements)
    launch_mt19937_blocks(
        out,
        state,
        scratch,
        output_elements=output_elements,
        rounds=rounds,
        block_count=block_count,
        output_mode=output_mode,
        requested_num_warps=num_warps,
    )
    generator._ws_next_block_start = (
        int(getattr(generator, "_ws_next_block_start", 0)) + rounds * block_count * MT19937_N
    )
