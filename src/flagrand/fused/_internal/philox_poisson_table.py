from __future__ import annotations

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.transforms import uint32_to_uniform


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _poisson_table_lookup(u, cdf_ptr, table_size, STEPS: tl.constexpr):
    lo = tl.full(u.shape, 0, tl.int32)
    hi = tl.full(u.shape, table_size - 1, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        cdf = tl.load(cdf_ptr + mid)
        left = u <= cdf
        hi = tl.where(left, mid, hi)
        lo = tl.where(left, lo, mid + 1)
    return lo


@triton.jit
def _philox_poisson_table_u32_kernel(
    out_ptr,
    cdf_ptr,
    table_size,
    seed,
    base_counter,
    n,
    n_counters,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u01 = tl.join(uint32_to_uniform(r0), uint32_to_uniform(r1))
    u23 = tl.join(uint32_to_uniform(r2), uint32_to_uniform(r3))
    uniforms = tl.reshape(tl.join(u01, u23), (BLOCK, 4))
    k = _poisson_table_lookup(uniforms, cdf_ptr, table_size, STEPS)

    tl.store(out_ptr + base, k, mask=mask[:, None] & (base < n))


def generate_philox_poisson_table_u32(
    out: torch.Tensor,
    *,
    seed: int,
    offset: int,
    lambda_val: float,
    block_size: int,
    num_warps: int,
) -> None:
    table = _cdf_table(lambda_val, str(out.device))
    n_counters = triton.cdiv(out.numel(), 4)
    grid = (triton.cdiv(n_counters, block_size),)
    _philox_poisson_table_u32_kernel[grid](
        out.view(-1),
        table,
        table.numel(),
        seed,
        offset // 4,
        out.numel(),
        n_counters,
        BLOCK=block_size,
        STEPS=math.ceil(math.log2(table.numel())),
        num_warps=num_warps,
    )


@lru_cache(maxsize=128)
def _cdf_table(lambda_val: float, device: str) -> torch.Tensor:
    return torch.tensor(_cdf_values(round(float(lambda_val), 12)), dtype=torch.float32, device=device)


@lru_cache(maxsize=128)
def _cdf_values(lambda_val: float) -> tuple[float, ...]:
    if lambda_val <= 0.0:
        raise ValueError(f"lambda must be positive, got {lambda_val}.")

    values: list[float] = []
    p = math.exp(-lambda_val)
    cdf = p
    k = 0
    values.append(cdf)
    while cdf < 1.0 - 1e-7:
        k += 1
        p *= lambda_val / k
        cdf += p
        values.append(min(cdf, 1.0))
        if k > lambda_val + 12.0 * math.sqrt(lambda_val) + 64.0:
            break
    values[-1] = 1.0
    return tuple(values)
