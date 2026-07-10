from __future__ import annotations

import torch

from flagrand.rng._mt19937_data import (
    MT19937_N,
    MT19937_NUM_STREAMS,
    MT19937_PREFETCH_BLOCKS,
    MT19937_SEQUENCE_CHUNK,
)
from flagrand.rng._mt19937_state import advance_to_block_start, generate_blocks_into
from flagrand.rng._sequence import clear_chunk_cache


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
    block_start_element = (current // MT19937_N) * MT19937_N
    advance_to_block_start(generator, block_start_element, device_str, num_warps)
    cache = torch.empty(MT19937_N, device=device, dtype=dtype)
    block_start = (block_start_element % MT19937_SEQUENCE_CHUNK) // MT19937_N
    generate_blocks_into(
        generator,
        cache,
        device_str,
        num_warps,
        block_start=block_start,
        block_count=1,
        rounds=1,
        output_mode=output_mode,
    )
    setattr(generator, "_chunk_cache", cache)
    setattr(generator, "_chunk_cache_start", block_start_element)
    setattr(generator, "_chunk_cache_key", cache_key)


def prefetch_blocks(
    generator,
    current: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_key: tuple[object, ...],
    device_str: str,
    num_warps: int,
    output_mode: int,
    *,
    multi_round: bool,
) -> None:
    block_start, block_count, rounds = resolve_prefetch_plan(
        current,
        multi_round=multi_round,
    )
    cache = torch.empty(rounds * block_count * MT19937_N, device=device, dtype=dtype)
    generate_blocks_into(
        generator,
        cache,
        device_str,
        num_warps,
        block_start=block_start,
        block_count=block_count,
        rounds=rounds,
        output_mode=output_mode,
    )
    setattr(generator, "_chunk_cache", cache)
    setattr(generator, "_chunk_cache_start", current)
    setattr(generator, "_chunk_cache_key", cache_key)


def resolve_prefetch_plan(current: int, *, multi_round: bool) -> tuple[int, int, int]:
    block_start = (current % MT19937_SEQUENCE_CHUNK) // MT19937_N
    prefetch_rounds = max(
        1,
        (MT19937_PREFETCH_BLOCKS + MT19937_NUM_STREAMS - 1) // MT19937_NUM_STREAMS,
    )
    if multi_round and block_start == 0 and prefetch_rounds > 1:
        return block_start, MT19937_NUM_STREAMS, prefetch_rounds
    block_count = min(MT19937_PREFETCH_BLOCKS, MT19937_NUM_STREAMS - block_start)
    return block_start, block_count, 1
