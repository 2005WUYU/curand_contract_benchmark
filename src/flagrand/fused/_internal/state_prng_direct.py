from __future__ import annotations

import torch

from flagrand.fused._internal.state_prng_kernels import (
    normal_kernel,
    poisson_large_kernel,
    poisson_small_kernel,
    uniform_kernel,
)
from flagrand.fused._internal.state_prng_state import RNG_MRG32K3A, RNG_XORWOW
from flagrand.runtime import CachedKernelLauncher

_BLOCK: int = 128
_TARGET_THREADS: int = 131072

_UNIFORM_LAUNCHER = CachedKernelLauncher(
    uniform_kernel,
    constexpr_names=("BLOCK", "RNG_KIND"),
)
_NORMAL_LAUNCHER = CachedKernelLauncher(
    normal_kernel,
    constexpr_names=("LOGNORMAL", "BLOCK", "RNG_KIND"),
)
_POISSON_SMALL_LAUNCHER = CachedKernelLauncher(
    poisson_small_kernel,
    constexpr_names=("BLOCK", "MAX_K", "RNG_KIND"),
)
_POISSON_LARGE_LAUNCHER = CachedKernelLauncher(
    poisson_large_kernel,
    constexpr_names=("BLOCK", "RNG_KIND"),
)


def generate_xorwow_uniform_f32(out: torch.Tensor, generator) -> None:
    _launch_uniform(out, generator, RNG_XORWOW, "generate_uniform")


def generate_xorwow_normal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, RNG_XORWOW, mean, stddev, lognormal=False, op_name="generate_normal")


def generate_xorwow_lognormal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, RNG_XORWOW, mean, stddev, lognormal=True, op_name="generate_lognormal")


def generate_xorwow_poisson_u32(out: torch.Tensor, generator, *, lambda_val: float, max_k: int) -> None:
    _launch_poisson(out, generator, RNG_XORWOW, lambda_val, max_k, "generate_poisson")


def generate_mrg32k3a_uniform_f32(out: torch.Tensor, generator) -> None:
    _launch_uniform(out, generator, RNG_MRG32K3A, "generate_uniform")


def generate_mrg32k3a_normal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, RNG_MRG32K3A, mean, stddev, lognormal=False, op_name="generate_normal")


def generate_mrg32k3a_lognormal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, RNG_MRG32K3A, mean, stddev, lognormal=True, op_name="generate_lognormal")


def generate_mrg32k3a_poisson_u32(out: torch.Tensor, generator, *, lambda_val: float, max_k: int) -> None:
    _launch_poisson(out, generator, RNG_MRG32K3A, lambda_val, max_k, "generate_poisson")


def _launch_uniform(out: torch.Tensor, generator, rng_kind: int, op_name: str) -> None:
    seed_lo, seed_hi, offset_val = _launch_seed_args(generator, op_name)
    n = out.numel()
    n_threads, num_iters = _thread_plan(n)
    grid = (_ceil_div(n_threads, _BLOCK),)
    _UNIFORM_LAUNCHER.launch(
        grid,
        (
            out,
            seed_lo,
            seed_hi,
            offset_val & 0xFFFFFFFF,
            n,
            n_threads,
            num_iters,
        ),
        (_BLOCK, rng_kind),
        (4,),
    )
    generator.offset = offset_val + n


def _launch_normal(
    out: torch.Tensor,
    generator,
    rng_kind: int,
    mean: float,
    stddev: float,
    *,
    lognormal: bool,
    op_name: str,
) -> None:
    seed_lo, seed_hi, offset_val = _launch_seed_args(generator, op_name)
    n_pairs = out.numel() // 2
    n_threads, num_iters = _thread_plan(n_pairs)
    grid = (_ceil_div(n_threads, _BLOCK),)
    _NORMAL_LAUNCHER.launch(
        grid,
        (
            out,
            seed_lo,
            seed_hi,
            offset_val & 0xFFFFFFFF,
            n_pairs,
            n_threads,
            num_iters,
            mean,
            stddev,
        ),
        (lognormal, _BLOCK, rng_kind),
        (4,),
    )
    generator.offset = offset_val + out.numel()


def _launch_poisson(
    out: torch.Tensor,
    generator,
    rng_kind: int,
    lambda_val: float,
    max_k: int,
    op_name: str,
) -> None:
    seed_lo, seed_hi, offset_val = _launch_seed_args(generator, op_name)
    if lambda_val < 30.0:
        n = out.numel()
        n_threads, num_iters = _thread_plan(n)
        grid = (_ceil_div(n_threads, _BLOCK),)
        _POISSON_SMALL_LAUNCHER.launch(
            grid,
            (
                out,
                seed_lo,
                seed_hi,
                offset_val & 0xFFFFFFFF,
                n,
                n_threads,
                num_iters,
                lambda_val,
            ),
            (_BLOCK, max_k, rng_kind),
            (4,),
        )
    else:
        n_pairs = out.numel() // 2
        n_threads, num_iters = _thread_plan(n_pairs)
        grid = (_ceil_div(n_threads, _BLOCK),)
        _POISSON_LARGE_LAUNCHER.launch(
            grid,
            (
                out,
                seed_lo,
                seed_hi,
                offset_val & 0xFFFFFFFF,
                n_pairs,
                n_threads,
                num_iters,
                lambda_val,
            ),
            (_BLOCK, rng_kind),
            (4,),
        )
    generator.offset = offset_val + out.numel()


def _thread_plan(n_work_items: int) -> tuple[int, int]:
    n_threads = min(_TARGET_THREADS, _ceil_div(n_work_items, _BLOCK) * _BLOCK)
    n_threads = max(n_threads, _BLOCK)
    num_iters = _ceil_div(n_work_items, n_threads)
    return n_threads, num_iters


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _launch_seed_args(generator, op_name: str) -> tuple[int, int, int]:
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"{op_name}: offset must be >= 0, got {offset_val}.")
    seed_val = int(getattr(generator, "seed", 0))
    return seed_val & 0xFFFFFFFF, (seed_val >> 32) & 0xFFFFFFFF, offset_val
