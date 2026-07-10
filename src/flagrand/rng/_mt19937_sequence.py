from __future__ import annotations

import torch

from flagrand.rng._mt19937_cache import cache_one_block, copy_from_cache, prefetch_blocks
from flagrand.rng._mt19937_data import (
    MT19937_N,
    MT19937_MAX_CHUNKS_PER_LAUNCH,
    MT19937_NUM_STREAMS,
    MT19937_PREFETCH_LIMIT,
    MT19937_SEQUENCE_CHUNK,
)
from flagrand.rng._mt19937_state import (
    advance_to_block_start,
    ensure_working_state,
    generate_blocks_into,
)


def generate_mt19937_contiguous(
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
        target_block_start = (current // MT19937_N) * MT19937_N
        if current > next_block_start:
            advance_to_block_start(generator, target_block_start, device_str, num_warps)
            next_block_start = int(getattr(generator, "_ws_next_block_start", 0))
        if current < next_block_start:
            raise RuntimeError(
                "MT19937 cannot rewind without a cached partial block. "
                f"current={current}, next_block_start={next_block_start}."
            )

        if current % MT19937_N != 0:
            cache_one_block(generator, current, out.device, out.dtype, cache_key, device_str, num_warps, output_mode)
            continue

        if remaining <= MT19937_PREFETCH_LIMIT:
            prefetch_blocks(
                generator,
                current,
                out.device,
                out.dtype,
                cache_key,
                device_str,
                num_warps,
                output_mode,
                multi_round=flat.numel() <= MT19937_PREFETCH_LIMIT,
            )
            continue

        copied = generate_direct_blocks(
            generator,
            flat,
            written,
            current,
            remaining,
            device_str,
            num_warps,
            output_mode,
        )
        written += copied
        current += copied
        remaining -= copied


def generate_direct_blocks(
    generator,
    flat: torch.Tensor,
    written: int,
    current: int,
    remaining: int,
    device_str: str,
    num_warps: int,
    output_mode: int,
) -> int:
    if current % MT19937_SEQUENCE_CHUNK == 0 and remaining >= MT19937_SEQUENCE_CHUNK:
        full_chunks = remaining // MT19937_SEQUENCE_CHUNK
        launch_chunks = min(full_chunks, MT19937_MAX_CHUNKS_PER_LAUNCH)
        span = launch_chunks * MT19937_SEQUENCE_CHUNK
        generate_blocks_into(
            generator,
            flat[written : written + span],
            device_str,
            num_warps,
            block_start=0,
            block_count=MT19937_NUM_STREAMS,
            rounds=launch_chunks,
            output_mode=output_mode,
        )
        return span

    full_blocks = remaining // MT19937_N
    block_start = (current % MT19937_SEQUENCE_CHUNK) // MT19937_N
    block_count = min(full_blocks, MT19937_NUM_STREAMS - block_start)
    if block_count == 0:
        return 0

    span = block_count * MT19937_N
    generate_blocks_into(
        generator,
        flat[written : written + span],
        device_str,
        num_warps,
        block_start=block_start,
        block_count=block_count,
        rounds=1,
        output_mode=output_mode,
    )
    return span
