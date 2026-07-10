from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.runtime import CachedKernelLauncher
from flagrand.fused._internal.transforms import (
    uint32_pair_to_uniform64_curand_compat,
    uniform_to_normal_trig_f64,
)


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _philox_uniform_f64_kernel(out_ptr, seed, base_counter, n, n_counters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = offs * 2

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)

    tile = tl.join(u0, u1)
    offsets = base[:, None] + tl.arange(0, 2)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _philox_normal_f64_kernel(
    out_ptr,
    seed,
    base_counter,
    n,
    n_counters,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = offs * 2

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)
    n0, n1 = uniform_to_normal_trig_f64(u0, u1)

    if STANDARD:
        y0 = n0
        y1 = n1
    else:
        y0 = mean + stddev * n0
        y1 = mean + stddev * n1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)

    tile = tl.join(y0, y1)
    offsets = base[:, None] + tl.arange(0, 2)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


_PHILOX_UNIFORM_F64_LAUNCHER = CachedKernelLauncher(
    _philox_uniform_f64_kernel,
    constexpr_names=("BLOCK",),
)
_PHILOX_NORMAL_F64_LAUNCHER = CachedKernelLauncher(
    _philox_normal_f64_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "BLOCK"),
)


def generate_philox_uniform_f64(
    out: torch.Tensor,
    generator,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    seed_val, offset_val, n_counters = _philox_launch_args_f64(out, generator, "generate_uniform")
    grid = triton.cdiv(n_counters, block_size)
    _PHILOX_UNIFORM_F64_LAUNCHER.launch(
        grid,
        (out, seed_val, offset_val // 4, out.numel(), n_counters),
        (block_size,),
        (num_warps,),
    )
    generator.offset = offset_val + 4 * n_counters


def generate_philox_normal_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
) -> None:
    _generate_philox_normal_like_f64(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=False,
        op_name="generate_normal",
    )


def generate_philox_lognormal_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
) -> None:
    _generate_philox_normal_like_f64(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=True,
        op_name="generate_lognormal",
    )


def _generate_philox_normal_like_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    lognormal: bool,
    op_name: str,
) -> None:
    seed_val, offset_val, n_counters = _philox_launch_args_f64(out, generator, op_name)
    grid = triton.cdiv(n_counters, block_size)
    _PHILOX_NORMAL_F64_LAUNCHER.launch(
        grid,
        (out, seed_val, offset_val // 4, out.numel(), n_counters, mean, stddev),
        (lognormal, mean == 0.0 and stddev == 1.0, block_size),
        (num_warps,),
    )
    generator.offset = offset_val + 4 * n_counters


def _philox_launch_args_f64(out: torch.Tensor, generator, op_name: str) -> tuple[int, int, int]:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError(f"{op_name}: output must be contiguous.")
    if n % 2 != 0:
        raise ValueError(f"{op_name}: Philox float64 path requires an even element count, got {n}.")
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"{op_name}: Philox offset must be >= 0, got {offset_val}.")
    if offset_val % 4 != 0:
        raise ValueError(
            f"{op_name}: Philox float64 offset must be aligned to a Philox4 counter, got {offset_val}."
        )
    return int(getattr(generator, "seed", 0)), offset_val, n // 2
