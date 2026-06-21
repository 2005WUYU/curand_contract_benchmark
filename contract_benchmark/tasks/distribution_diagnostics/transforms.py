from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from contract_benchmark.adapters import flagrand_generate_raw, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.tasks.common import dtype_for_distribution
from contract_benchmark.tasks.distribution_diagnostics.cases import GENERATOR, DistributionCase, lambda_value

BLOCK_SIZE = 512
NUM_WARPS = 4


def prepare_raw_and_output(ctx: BenchmarkContext, case: DistributionCase, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.empty(n, device=ctx.device, dtype=torch.int32)
    out = torch.empty(n, device=ctx.device, dtype=dtype_for_distribution(case.distribution))
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)
    flagrand_generate_raw(raw, gen)
    torch.cuda.synchronize()
    return raw, out


def launch_transform(case: DistributionCase, raw: torch.Tensor, out: torch.Tensor, n: int) -> torch.Tensor:
    kernels = transform_kernels()
    triton = kernels["triton"]

    if case.distribution == "uniform_f32":
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kernels["uniform32"][grid](out.view(-1), raw.view(-1), n, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
    elif case.distribution == "normal_f32":
        launch_pair_transform(kernels["normal32"], triton, raw, out, n, 0.0, 1.0)
    elif case.distribution == "lognormal_f32":
        launch_pair_transform(kernels["lognormal32"], triton, raw, out, n, 0.0, 1.0)
    elif case.distribution == "poisson_u32":
        launch_poisson_transform(kernels, triton, case, raw, out, n)
    else:
        raise ValueError(f"Unsupported diagnostic distribution: {case.distribution}")
    return out


def launch_pair_transform(kernel: Any, triton: Any, raw: torch.Tensor, out: torch.Tensor, n: int, mean: float, stddev: float) -> None:
    n_pairs = n // 2
    grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
    kernel[grid](out.view(-1), raw.view(-1), n_pairs, mean, stddev, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)


def launch_poisson_transform(kernels: dict[str, Any], triton: Any, case: DistributionCase, raw: torch.Tensor, out: torch.Tensor, n: int) -> None:
    lambda_val = lambda_value(case)
    if lambda_val < 30.0:
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kernels["poisson_small32"][grid](out.view(-1), raw.view(-1), n, lambda_val, BLOCK=BLOCK_SIZE, MAX_K=192, num_warps=NUM_WARPS)
    else:
        n_pairs = n // 2
        grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
        kernels["poisson_large32"][grid](out.view(-1), raw.view(-1), n_pairs, lambda_val, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)


@lru_cache(maxsize=1)
def transform_kernels() -> dict[str, Any]:
    import triton
    from flagrand.fused.lognormal import _lognormal_transform_kernel_32
    from flagrand.fused.normal import _normal_transform_kernel_32
    from flagrand.fused.poisson import _poisson_transform_kernel_large_32, _poisson_transform_kernel_small_32
    from flagrand.fused.uniform import _uniform_transform_kernel_32

    return {
        "triton": triton,
        "uniform32": _uniform_transform_kernel_32,
        "normal32": _normal_transform_kernel_32,
        "lognormal32": _lognormal_transform_kernel_32,
        "poisson_small32": _poisson_transform_kernel_small_32,
        "poisson_large32": _poisson_transform_kernel_large_32,
    }
