from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, c2, c3)
    return r0, r1, r2, r3


@triton.jit
def _philox_kernel(out_ptr, seed, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    counter = offs
    r0, r1, r2, r3 = _philox_generate(seed, counter)
    # Build a (BLOCK, 4) tile and store with one 2D op so the compiler can
    # emit 16-byte vector stores (4 contiguous uint32 per counter).
    r01 = tl.join(r0, r1)            # (BLOCK, 2)
    r23 = tl.join(r2, r3)            # (BLOCK, 2)
    tile = tl.join(r01, r23)         # (BLOCK, 2, 2)
    tile = tl.reshape(tile, (BLOCK, 4))
    base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + base, tile.to(tl.int32, bitcast=True), mask=mask[:, None])


@dataclass
class PhiloxGenerator:
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
        block_size = kwargs.get("block_size", 512)
        num_warps = kwargs.get("num_warps", 4)

        n = out.numel()
        if n % 4 != 0:
            raise ValueError(
                f"Philox: element count must be a multiple of 4, got {n}."
            )

        n_counters = n // 4
        grid = (triton.cdiv(n_counters, block_size),)
        _philox_kernel[grid](
            out.view(-1),
            self.seed,
            n_counters,
            BLOCK=block_size,
            num_warps=num_warps,
        )
        return out
