from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from flagrand.rng._sequence import clear_chunk_cache, generate_chunked


# Multi-stream MT19937: NUM_STREAMS independent stream instances run in parallel.
# Each is a legal 624-state MT19937, seeded by splitmix32(seed XOR stream_id) and
# initialized via the standard MT19937 init recurrence. This differs from cuRAND
# (which uses jump-ahead by 2^1000 to slice one master sequence into 8192 segments,
# requiring offline tables) but is mathematically valid: each stream by itself is
# a real MT19937 with a deterministic state.
MT19937_N: int = 624
MT19937_M: int = 397
MT19937_INIT_MULT: int = 1812433253
NUM_STREAMS: int = 8192
SEQUENCE_CHUNK: int = NUM_STREAMS * MT19937_N


@triton.jit
def _mt19937_temper(y):
    full_mask = ~tl.zeros_like(y)
    y = y ^ ((y >> 11) & full_mask)
    y = y ^ ((y << 7) & 0x9D2C5680)
    y = y ^ ((y << 15) & 0xEFC60000)
    y = y ^ (y >> 18)
    return y


@triton.jit
def _mt19937_multistream_kernel(
    out_ptr,
    state_ptr,         # (NUM_STREAMS * N,) uint32, persistent state per stream
    scratch_ptr,       # (NUM_STREAMS * N,) uint32, scratch for in-block barrier
    n_total,
    n_rounds,          # how many twist rounds this call performs
    NUM_STREAMS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_STATE: tl.constexpr,  # >= N, power-of-2
    M: tl.constexpr,
):
    # 1 program = 1 stream. 8192 programs run in parallel across all SMs.
    stream_id = tl.program_id(0)
    state_base = stream_id * N
    scratch_base = stream_id * N

    tid = tl.arange(0, BLOCK_STATE)
    state_mask = tid < N

    state = tl.load(state_ptr + state_base + tid, mask=state_mask, other=0)

    upper_mask = tl.full((), 0x80000000, tl.uint32)
    lower_mask = tl.full((), 0x7FFFFFFF, tl.uint32)
    matrix_a = tl.full((), 0x9908B0DF, tl.uint32)
    one_u32 = tl.full((), 1, tl.uint32)
    zero_u32 = tl.full((), 0, tl.uint32)

    next_idx = tl.where(tid + 1 == N, 0, tid + 1)
    m_idx = tl.where(tid + M >= N, tid + M - N, tid + M)

    for k in range(n_rounds):
        # twist: vectorized over 624 lanes within this program. Use scratch
        # buffer + debug_barrier for cross-warp gather of state[next_idx]
        # and state[m_idx] (the same lane stores `state` then loads from the
        # rotated index, so this is a tight in-program shuffle).
        tl.store(scratch_ptr + scratch_base + tid, state, mask=state_mask)
        tl.debug_barrier()
        s_next = tl.load(scratch_ptr + scratch_base + next_idx, mask=state_mask, other=0)
        s_m = tl.load(scratch_ptr + scratch_base + m_idx, mask=state_mask, other=0)

        y = (state & upper_mask) | (s_next & lower_mask)
        mag = tl.where((y & one_u32) != zero_u32, matrix_a, zero_u32)
        state = s_m ^ (y >> 1) ^ mag

        tempered = _mt19937_temper(state)
        # Output layout: round k, stream s, lane j → out[(k * NUM_STREAMS + s) * N + j]
        out_offs = (k * NUM_STREAMS + stream_id) * N + tid
        out_mask = state_mask & (out_offs < n_total)
        tl.store(out_ptr + out_offs, tempered, mask=out_mask)

    tl.store(state_ptr + state_base + tid, state, mask=state_mask)


def _build_initial_states_cpu(seed: int) -> torch.Tensor:
    """Build (NUM_STREAMS, N) int32 initial state on CPU. Each stream uses
    splitmix32(seed XOR stream_id) as its seed for the standard MT19937 init.
    """
    def _splitmix32_py(x: int) -> int:
        x = x & 0xFFFFFFFF
        x = (x ^ (x >> 16)) & 0xFFFFFFFF
        x = (x * 0x85EBCA6B) & 0xFFFFFFFF
        x = (x ^ (x >> 13)) & 0xFFFFFFFF
        x = (x * 0xC2B2AE35) & 0xFFFFFFFF
        x = (x ^ (x >> 16)) & 0xFFFFFFFF
        return x

    seed_u32 = seed & 0xFFFFFFFF
    flat = [0] * (NUM_STREAMS * MT19937_N)
    for sid in range(NUM_STREAMS):
        base = _splitmix32_py(seed_u32 + sid)
        if base == 0:
            base = 1
        off = sid * MT19937_N
        flat[off] = base
        prev = base
        for i in range(1, MT19937_N):
            cur = (MT19937_INIT_MULT * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
            flat[off + i] = cur
            prev = cur

    t = torch.tensor(flat, dtype=torch.int64)
    t = torch.where(t >= 0x80000000, t - 0x100000000, t).to(torch.int32)
    return t.reshape(NUM_STREAMS, MT19937_N).contiguous()


@dataclass
class Mt19937Generator:
    seed: int = 0
    offset: int = 0

    _state: torch.Tensor | None = None
    _scratch: torch.Tensor | None = None
    _device: torch.device | None = None
    _initialized: bool = False

    @property
    def dimensions(self) -> None:
        return None

    def _ensure_initialized(self, device: torch.device) -> None:
        if not self._initialized or self._device != device:
            init = _build_initial_states_cpu(self.seed)
            self._state = init.to(device)
            self._scratch = torch.empty_like(self._state)
            self._device = device
            self._initialized = True
            clear_chunk_cache(self)

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        n = out.numel()
        if n == 0:
            return out

        if seed is not None:
            raise ValueError("MT19937 explicit seed override is not supported.")
        offset_val = self.offset if offset is None else int(offset)
        if offset is not None and offset_val != self.offset:
            raise ValueError("MT19937 explicit offset override is not supported.")
        if offset_val != 0 and not self._initialized:
            raise ValueError(f"MT19937 does not support non-zero initial offset, got {offset_val}.")

        device = out.device
        self._ensure_initialized(device)

        generate_chunked(
            self,
            out,
            start=offset_val,
            chunk_size=SEQUENCE_CHUNK,
            cache_key=(self.seed, str(out.device), str(out.dtype)),
            generate_chunk=lambda chunk, chunk_start: self._generate_chunk(chunk),
        )
        self.offset = offset_val + n
        return out

    def _generate_chunk(self, out: torch.Tensor) -> None:
        n = out.numel()
        n_rounds = (n + SEQUENCE_CHUNK - 1) // SEQUENCE_CHUNK

        state_u32 = self._state.view(-1).view(torch.uint32)
        scratch_u32 = self._scratch.view(-1).view(torch.uint32)
        out_u32 = out.view(-1).view(torch.uint32)

        _mt19937_multistream_kernel[(NUM_STREAMS,)](
            out_u32,
            state_u32,
            scratch_u32,
            n,
            n_rounds,
            NUM_STREAMS=NUM_STREAMS,
            N=MT19937_N,
            BLOCK_STATE=1024,
            M=MT19937_M,
            num_warps=4,
        )
