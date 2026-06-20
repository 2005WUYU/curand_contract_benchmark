from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


_BLOCK: int = 256
_TARGET_THREADS: int = 4096


@triton.jit
def _splitmix32(x):
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _xorwow_step(v0, v1, v2, v3, v4, d):
    t = v0 ^ (v0 >> 2)
    nv0, nv1, nv2, nv3 = v1, v2, v3, v4
    nv4 = (v4 ^ (v4 << 4)) ^ (t ^ (t << 1))
    nd = d + 362437  # Weyl delta
    output = nv4 + nd
    return output, nv0, nv1, nv2, nv3, nv4, nd


@triton.jit
def _xorwow_init_per_thread(seed_lo, seed_hi, tid, offset_u32):
    tid_u = tid.to(tl.uint32)
    sl = seed_lo ^ _splitmix32(tid_u + offset_u32)
    sh = seed_hi ^ _splitmix32(tid_u + offset_u32 + 0x9E3779B9)

    # Marsaglia 2003 init: Multipliers and additive constants from cuRAND-style derivation
    t0 = 1099087573 * sl
    t1 = 2591861531 * sh

    d = 6615241 + t1 + t0
    v0 = 123456789 + t0
    v1 = 362436069 ^ t0
    v2 = 521288629 + t1
    v3 = 88675123 ^ t1
    v4 = 5783321 + t0
    return v0, v1, v2, v3, v4, d


@triton.jit
def _xorwow_kernel(
    out_ptr,
    seed_lo,
    seed_hi,
    offset_u32,
    n,
    n_threads,
    num_iters,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    thread_mask = tid < n_threads

    v0, v1, v2, v3, v4, d = _xorwow_init_per_thread(seed_lo, seed_hi, tid, offset_u32)

    for k in range(num_iters):
        output, v0, v1, v2, v3, v4, d = _xorwow_step(v0, v1, v2, v3, v4, d)
        out_offs = k * n_threads + tid
        out_mask = thread_mask & (out_offs < n)
        tl.store(out_ptr + out_offs, output, mask=out_mask)


@dataclass
class XorwowGenerator:
    seed: int = 0
    offset: int = 0

    @property
    def dimensions(self) -> None:
        return None

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

        n_threads = min(_TARGET_THREADS, ((n + _BLOCK - 1) // _BLOCK) * _BLOCK)
        n_threads = max(n_threads, _BLOCK)
        num_iters = (n + n_threads - 1) // n_threads
        grid = ((n_threads + _BLOCK - 1) // _BLOCK,)

        seed_lo = self.seed & 0xFFFFFFFF
        seed_hi = (self.seed >> 32) & 0xFFFFFFFF

        out_u32 = out.view(-1).view(torch.uint32)

        _xorwow_kernel[grid](
            out_u32,
            seed_lo,
            seed_hi,
            self.offset & 0xFFFFFFFF,
            n,
            n_threads,
            num_iters,
            BLOCK=_BLOCK,
            num_warps=8,
        )
        return out
