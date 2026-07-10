from __future__ import annotations

import torch

from flagrand.rng._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_SEQUENCE_CHUNK
from flagrand.rng._mtgp32_state import advance_to_block_start, generate_blocks_into
from flagrand.rng._sequence import clear_chunk_cache
from flagrand.rng._stateful_output import StatefulOutput


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
    cache_valid = (
        cache is not None
        and getattr(generator, "_chunk_cache_key", None) == cache_key
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
    output: StatefulOutput,
) -> None:
    block_start_element = (current // MTGP32_BLOCK_SIZE) * MTGP32_BLOCK_SIZE
    advance_to_block_start(generator, block_start_element, device_str, num_warps)
    cache = torch.empty(MTGP32_BLOCK_SIZE, device=device, dtype=dtype)
    block_start = (block_start_element % MTGP32_SEQUENCE_CHUNK) // MTGP32_BLOCK_SIZE
    generator._chunk_cache_state_snapshot = generator._working_state[block_start].clone()
    generator._chunk_cache_state_index = block_start
    generator._chunk_cache_ws_start = int(getattr(generator, "_ws_next_block_start", 0))
    generate_blocks_into(
        generator,
        cache,
        device_str,
        num_warps,
        block_start=block_start,
        block_count=1,
        chunks=1,
        output=output,
    )
    generator._chunk_cache = cache
    generator._chunk_cache_start = block_start_element
    generator._chunk_cache_key = cache_key


def restore_partial_block_for_output(
    generator,
    current: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_key: tuple[object, ...],
    device_str: str,
    num_warps: int,
    output: StatefulOutput,
) -> bool:
    cache = getattr(generator, "_chunk_cache", None)
    cache_start = int(getattr(generator, "_chunk_cache_start", -1))
    if (
        cache is None
        or getattr(generator, "_chunk_cache_key", None) == cache_key
        or not cache_start <= current < cache_start + cache.numel()
    ):
        return False

    state_index = int(generator._chunk_cache_state_index)
    generator._working_state[state_index].copy_(generator._chunk_cache_state_snapshot)
    generator._ws_next_block_start = int(generator._chunk_cache_ws_start)
    clear_chunk_cache(generator)
    cache_one_block(
        generator,
        current,
        device,
        dtype,
        cache_key,
        device_str,
        num_warps,
        output,
    )
    return True
