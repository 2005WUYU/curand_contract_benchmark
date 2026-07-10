from __future__ import annotations

import torch


# Multi-stream MT19937: NUM_STREAMS independent stream instances run in parallel.
# Each stream is a legal 624-state MT19937 initialized with splitmix32(seed + id).
MT19937_N: int = 624
MT19937_M: int = 397
MT19937_INIT_MULT: int = 1812433253
MT19937_NUM_STREAMS: int = 3072
MT19937_SEQUENCE_CHUNK: int = MT19937_NUM_STREAMS * MT19937_N
MT19937_MAX_CHUNKS_PER_LAUNCH: int = 8
MT19937_PREFETCH_BLOCKS: int = 4096
MT19937_PREFETCH_LIMIT: int = 2_000_000


def build_initial_states(seed: int) -> torch.Tensor:
    seed_u32 = seed & 0xFFFFFFFF
    flat = [0] * (MT19937_NUM_STREAMS * MT19937_N)
    for sid in range(MT19937_NUM_STREAMS):
        base = splitmix32(seed_u32 + sid)
        if base == 0:
            base = 1
        off = sid * MT19937_N
        flat[off] = base
        prev = base
        for i in range(1, MT19937_N):
            cur = (MT19937_INIT_MULT * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
            flat[off + i] = cur
            prev = cur

    state = torch.tensor(flat, dtype=torch.int64)
    state = torch.where(state >= 0x80000000, state - 0x100000000, state).to(torch.int32)
    return state.reshape(MT19937_NUM_STREAMS, MT19937_N).contiguous()


def splitmix32(x: int) -> int:
    x = x & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x
