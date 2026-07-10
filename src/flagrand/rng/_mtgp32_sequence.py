from __future__ import annotations

import torch

from flagrand.rng._mtgp32_aligned_kernel import generate_aligned_mtgp32
from flagrand.rng._mtgp32_cache import (
    cache_one_block,
    copy_from_cache,
    restore_partial_block_for_output,
)
from flagrand.rng._mtgp32_data import (
    MTGP32_BLOCK_SIZE,
    MTGP32_MAX_BLOCKS,
    MTGP32_MAX_CHUNKS_PER_LAUNCH,
    MTGP32_SEQUENCE_CHUNK,
)
from flagrand.rng._mtgp32_state import (
    advance_to_block_start,
    ensure_working_state,
    generate_blocks_into,
)
from flagrand.rng._stateful_output import RAW_OUTPUT, StatefulOutput


def generate_mtgp32_contiguous(
    generator,
    out: torch.Tensor,
    seed_val: int,
    offset_val: int,
    num_warps: int,
    *,
    output: StatefulOutput = RAW_OUTPUT,
) -> None:
    flat = out if out.ndim == 1 else out.view(-1)
    device_str = str(out.device)
    ensure_working_state(generator, seed_val, device_str)

    if (
        offset_val % MTGP32_BLOCK_SIZE == 0
        and flat.numel() % MTGP32_BLOCK_SIZE == 0
        and flat.numel() >= MTGP32_SEQUENCE_CHUNK
    ):
        generate_aligned_mtgp32(
            generator,
            flat,
            block_offset=offset_val // MTGP32_BLOCK_SIZE,
            num_warps=num_warps,
            output=output,
        )
        return

    cache_key = (seed_val, device_str, str(out.dtype), num_warps, output.cache_key)
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

        if restore_partial_block_for_output(
            generator,
            current,
            out.device,
            out.dtype,
            cache_key,
            device_str,
            num_warps,
            output,
        ):
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
            cache_one_block(
                generator,
                current,
                out.device,
                out.dtype,
                cache_key,
                device_str,
                num_warps,
                output,
            )
            continue

        if current % MTGP32_SEQUENCE_CHUNK == 0 and remaining >= MTGP32_SEQUENCE_CHUNK:
            launch_chunks = min(
                remaining // MTGP32_SEQUENCE_CHUNK,
                MTGP32_MAX_CHUNKS_PER_LAUNCH,
            )
            span = launch_chunks * MTGP32_SEQUENCE_CHUNK
            generate_blocks_into(
                generator,
                flat[written : written + span],
                device_str,
                num_warps,
                block_start=0,
                block_count=MTGP32_MAX_BLOCKS,
                chunks=launch_chunks,
                output=output,
            )
            written += span
            current += span
            remaining -= span
            continue

        block_start = (current % MTGP32_SEQUENCE_CHUNK) // MTGP32_BLOCK_SIZE
        block_count = min(
            remaining // MTGP32_BLOCK_SIZE,
            MTGP32_MAX_BLOCKS - block_start,
        )
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
                output=output,
            )
            written += span
            current += span
            remaining -= span
            continue

        cache_one_block(
            generator,
            current,
            out.device,
            out.dtype,
            cache_key,
            device_str,
            num_warps,
            output,
        )
