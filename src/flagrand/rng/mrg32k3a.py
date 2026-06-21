from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from flagrand.rng._sequence import generate_chunked

_BLOCK: int = 128
_TARGET_THREADS: int = 131072
_SEQUENCE_CHUNK: int = 1 << 20


@triton.jit
def _splitmix32(x):
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _mrg32k3a_step(s1_0, s1_1, s1_2, s2_0, s2_1, s2_2):
    x1 = (1403580 * s1_1 + 4294156359 * s1_2) % 4294967087
    x2 = (527612 * s2_0 + 4293573854 * s2_2) % 4294944443
    # output = (x1 - x2) mod M1, with x1 in [0, M1) and x2 in [0, M2);
    # diff in [-(M2-1), M1-1] so a single conditional add brings it into [0, M1)
    diff = x1 - x2
    output = tl.where(diff < 0, diff + 4294967087, diff)
    return output, s1_1, s1_2, x1, s2_1, s2_2, x2


@triton.jit
def _mrg32k3a_init_per_thread(seed_u32, tid, offset_u32):
    tid_u = tid.to(tl.uint32)
    pert = _splitmix32(seed_u32 + tid_u + offset_u32)
    pert_b = _splitmix32(seed_u32 + tid_u + offset_u32 + 0x9E3779B9)

    pert64 = pert.to(tl.int64)
    pert_b64 = pert_b.to(tl.int64)

    s1_0 = (123456789 + pert64) % 4294967087
    s1_1 = (362436069 + pert_b64) % 4294967087
    s1_2 = (521288629 + pert64 + pert_b64) % 4294967087
    s2_0 = (88675123 + pert64) % 4294944443
    s2_1 = (5783321 + pert_b64) % 4294944443
    s2_2 = (6615241 + pert64 + pert_b64) % 4294944443

    # state must be non-zero
    s1_0 = tl.where(s1_0 == 0, 1, s1_0)
    s1_1 = tl.where(s1_1 == 0, 1, s1_1)
    s1_2 = tl.where(s1_2 == 0, 1, s1_2)
    s2_0 = tl.where(s2_0 == 0, 1, s2_0)
    s2_1 = tl.where(s2_1 == 0, 1, s2_1)
    s2_2 = tl.where(s2_2 == 0, 1, s2_2)
    return s1_0, s1_1, s1_2, s2_0, s2_1, s2_2


@triton.jit
def _mrg32k3a_kernel(
    out_ptr,
    seed_u32,
    offset_u32,
    n,
    n_threads,
    num_iters,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    thread_mask = tid < n_threads

    s1_0, s1_1, s1_2, s2_0, s2_1, s2_2 = _mrg32k3a_init_per_thread(
        seed_u32, tid, offset_u32,
    )

    for k in range(num_iters):
        output, s1_0, s1_1, s1_2, s2_0, s2_1, s2_2 = _mrg32k3a_step(
            s1_0, s1_1, s1_2, s2_0, s2_1, s2_2,
        )
        # Output is in [0, M1), truncate to uint32 then bitcast for storage
        output_u32 = (output & 0xFFFFFFFF).to(tl.uint32)
        out_offs = k * n_threads + tid
        out_mask = thread_mask & (out_offs < n)
        tl.store(out_ptr + out_offs, output_u32.to(tl.int32, bitcast=True), mask=out_mask)


@dataclass
class Mrg32k3aGenerator:
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

        seed_val = self.seed if seed is None else int(seed)
        offset_val = self.offset if offset is None else int(offset)
        if offset_val < 0:
            raise ValueError(f"MRG32K3A: offset must be >= 0, got {offset_val}.")

        if seed is not None or offset is not None:
            _launch_mrg32k3a(out, seed_val, offset_val)
            return out

        generate_chunked(
            self,
            out,
            start=offset_val,
            chunk_size=_SEQUENCE_CHUNK,
            cache_key=(seed_val, str(out.device), str(out.dtype)),
            generate_chunk=lambda chunk, chunk_start: _launch_mrg32k3a(chunk, seed_val, chunk_start),
        )
        self.offset = offset_val + n
        return out


def _launch_mrg32k3a(out: torch.Tensor, seed_val: int, offset_val: int) -> None:
    n = out.numel()
    if n == 0:
        return

    n_threads = min(_TARGET_THREADS, ((n + _BLOCK - 1) // _BLOCK) * _BLOCK)
    n_threads = max(n_threads, _BLOCK)
    num_iters = (n + n_threads - 1) // n_threads
    grid = ((n_threads + _BLOCK - 1) // _BLOCK,)

    _mrg32k3a_kernel[grid](
        out.view(-1),
        seed_val & 0xFFFFFFFF,
        offset_val & 0xFFFFFFFF,
        n,
        n_threads,
        num_iters,
        BLOCK=_BLOCK,
        num_warps=4,
    )
