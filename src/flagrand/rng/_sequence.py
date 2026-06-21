from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


ChunkGenerator = Callable[[torch.Tensor, int], None]


def clear_chunk_cache(owner: object) -> None:
    for name in ("_chunk_cache", "_chunk_cache_start", "_chunk_cache_key"):
        if hasattr(owner, name):
            delattr(owner, name)


def generate_chunked(
    owner: object,
    out: torch.Tensor,
    *,
    start: int,
    chunk_size: int,
    cache_key: tuple[Any, ...],
    generate_chunk: ChunkGenerator,
) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

    flat = out.view(-1)
    written = 0
    current = int(start)
    remaining = flat.numel()

    while remaining:
        cache = getattr(owner, "_chunk_cache", None)
        cache_start = int(getattr(owner, "_chunk_cache_start", -1))
        cache_key_current = getattr(owner, "_chunk_cache_key", None)
        cache_valid = (
            cache is not None
            and cache_key_current == cache_key
            and cache_start <= current < cache_start + cache.numel()
        )

        if cache_valid:
            cache_offset = current - cache_start
            take = min(remaining, cache.numel() - cache_offset)
            flat[written : written + take].copy_(cache[cache_offset : cache_offset + take])
            written += take
            current += take
            remaining -= take
            if cache_offset + take == cache.numel():
                clear_chunk_cache(owner)
            continue

        if current % chunk_size == 0 and remaining >= chunk_size:
            generate_chunk(flat[written : written + chunk_size], current)
            written += chunk_size
            current += chunk_size
            remaining -= chunk_size
            continue

        chunk_start = (current // chunk_size) * chunk_size
        cache = torch.empty(chunk_size, device=out.device, dtype=out.dtype)
        generate_chunk(cache, chunk_start)
        setattr(owner, "_chunk_cache", cache)
        setattr(owner, "_chunk_cache_start", chunk_start)
        setattr(owner, "_chunk_cache_key", cache_key)
