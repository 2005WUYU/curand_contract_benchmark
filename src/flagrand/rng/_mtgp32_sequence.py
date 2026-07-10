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


def generate_mtgp32_contiguous(
    generator,
    out: torch.Tensor,
    seed_val: int,
    offset_val: int,
    num_warps: int,
    *,
    output_mode: int = 0,
) -> None:
    flat = out.view(-1)
    device_str = str(out.device)
    cache_key = (seed_val, device_str, str(out.dtype), num_warps, output_mode)
    ensure_working_state(generator, seed_val, device_str)

    written = 0
    current = int(offset_val)
    remaining = flat.numel()
    while remaining:
        copied = copy_from_cache(generator, flat, written, current, remaining, cache_key)
        if copied:
            written += copied
            current += copied
            remaining -= copied
            continue

        next_block_start = int(getattr(generator, "_ws_next_block_start", 0))
        target_block_start = (current // MTGP32_BLOCK_SIZE) * MTGP32_BLOCK_SIZE
        if current > next_block_start:
            advance_to_block_start(generator, target_block_start, device_str, num_warps)
            next_block_start = int(getattr(generator, "_ws_next_block_start", 0))
        if current < next_block_start:
            raise RuntimeError(
                "MTGP32 cannot rewind without a cached partial block. "
                f"current={current}, next_block_start={next_block_start}."
            )

        if current % MTGP32_BLOCK_SIZE != 0:
            cache_one_block(generator, current, out.device, out.dtype, cache_key, device_str, num_warps, output_mode)
            continue

        if current % MTGP32_SEQUENCE_CHUNK == 0 and remaining >= MTGP32_SEQUENCE_CHUNK:
            full_chunks = remaining // MTGP32_SEQUENCE_CHUNK
            launch_chunks = min(full_chunks, MTGP32_MAX_CHUNKS_PER_LAUNCH)
            span = launch_chunks * MTGP32_SEQUENCE_CHUNK
            generate_blocks_into(
                generator,
                flat[written : written + span],
                device_str,
                num_warps,
                block_start=0,
                block_count=MTGP32_MAX_BLOCKS,
                chunks=launch_chunks,
                output_mode=output_mode,
            )
            written += span
            current += span
            remaining -= span
            continue

        full_blocks = remaining // MTGP32_BLOCK_SIZE
        block_start = (current % MTGP32_SEQUENCE_CHUNK) // MTGP32_BLOCK_SIZE
        block_count = min(full_blocks, MTGP32_MAX_BLOCKS - block_start)
        if block_count:
            span = block_count * MTGP32_BLOCK_SIZE
            generate_blocks_into(
                generator,
                flat[written : written + span],
                device_str,
                num_warps,
                block_start=block_start,
                block_count=block_count,
                chunks=1,
                output_mode=output_mode,
            )
            written += span
            current += span
            remaining -= span
            continue

        cache_one_block(generator, current, out.device, out.dtype, cache_key, device_str, num_warps, output_mode)


def ensure_working_state(generator, seed_val: int, device_str: str) -> None:
    ws_seed = getattr(generator, "_ws_seed", None)
    ws_device = getattr(generator, "_ws_device", None)
    ws_blocks = getattr(generator, "_ws_blocks", 0)
    if ws_seed == seed_val and ws_device == device_str and ws_blocks >= MTGP32_MAX_BLOCKS:
        return
    initial_state = build_initial_state(seed_val, device_str)
    generator._working_state = initial_state.clone()
    generator._ws_seed = seed_val
    generator._ws_device = device_str
    generator._ws_blocks = generator._working_state.shape[0]
    generator._ws_next_block_start = 0
    clear_chunk_cache(generator)


def copy_from_cache(
    generator,
    flat: torch.Tensor,
    written: int,
    current: int,
    remaining: int,
    cache_key: tuple[object, ...],
) -> int:
    cache = getattr(generator, "_chunk_cache", None)
    cache_start = int(getattr(generator, "_chunk_cache_start", -1))
    cache_key_current = getattr(generator, "_chunk_cache_key", None)
    cache_valid = (
        cache is not None
        and cache_key_current == cache_key
        and cache_start <= current < cache_start + cache.numel()
    )
    if not cache_valid:
        return 0
    cache_offset = current - cache_start
    take = min(remaining, cache.numel() - cache_offset)
    flat[written : written + take].copy_(cache[cache_offset : cache_offset + take])
    if cache_offset + take == cache.numel():
        clear_chunk_cache(generator)
    return take


def cache_one_block(
    generator,
    current: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_key: tuple[object, ...],
    device_str: str,
    num_warps: int,
    output_mode: int,
) -> None:
    block_start_element = (current // MTGP32_BLOCK_SIZE) * MTGP32_BLOCK_SIZE
    advance_to_block_start(generator, block_start_element, device_str, num_warps)
    cache = torch.empty(MTGP32_BLOCK_SIZE, device=device, dtype=dtype)
    block_start = (block_start_element % MTGP32_SEQUENCE_CHUNK) // MTGP32_BLOCK_SIZE
    generate_blocks_into(
        generator,
        cache,
        device_str,
        num_warps,
        block_start=block_start,
        block_count=1,
        chunks=1,
        output_mode=output_mode,
    )
    setattr(generator, "_chunk_cache", cache)
    setattr(generator, "_chunk_cache_start", block_start_element)
    setattr(generator, "_chunk_cache_key", cache_key)


def advance_to_block_start(generator, block_start_element: int, device_str: str, num_warps: int) -> None:
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
            launch_chunks = min(blocks_to_skip // MTGP32_MAX_BLOCKS, MTGP32_MAX_CHUNKS_PER_LAUNCH)
            generate_blocks_into(
                generator,
                scratch,
                device_str,
                num_warps,
                block_start=0,
                block_count=MTGP32_MAX_BLOCKS,
                chunks=launch_chunks,
                n_elements=0,
                output_mode=0,
            )
            skipped = launch_chunks * MTGP32_MAX_BLOCKS
        else:
            block_count = min(blocks_to_skip, MTGP32_MAX_BLOCKS - block_start)
            generate_blocks_into(
                generator,
                scratch,
                device_str,
                num_warps,
                block_start=block_start,
                block_count=block_count,
                chunks=1,
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
    chunks: int,
    n_elements: int | None = None,
    output_mode: int = 0,
) -> None:
    if chunks <= 0 or block_count <= 0:
        return
    if chunks > 1 and (block_start != 0 or block_count != MTGP32_MAX_BLOCKS):
        raise ValueError("MTGP32 multi-chunk launch requires a full 192-block chunk.")
    pos, sh1, sh2, param, temper = build_param_tensors(device_str)
    block_end = block_start + block_count
    state = generator._working_state[block_start:block_end]
    start_iter = (int(getattr(generator, "_ws_next_block_start", 0)) // MTGP32_SEQUENCE_CHUNK) % 4
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
        output_mode=output_mode,
        requested_num_warps=num_warps,
    )
    generator._ws_next_block_start = (
        int(getattr(generator, "_ws_next_block_start", 0)) + chunks * block_count * MTGP32_BLOCK_SIZE
    )
