from __future__ import annotations

import torch

from contract_benchmark.curand_generator import CurandGenerator


def make_curand_generator(
    generator: str,
    *,
    seed: int,
    offset: int = 0,
    ordering: str | None = "legacy",
    dimensions: int | None = None,
) -> CurandGenerator:
    return CurandGenerator(
        generator,
        seed=seed,
        offset=offset,
        ordering=ordering,
        dimensions=dimensions,
    )


def curand_generate_by_distribution(
    gen: CurandGenerator,
    out: torch.Tensor,
    distribution: str,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
    lambda_val: float = 10.0,
) -> torch.Tensor:
    if distribution == "raw32":
        return gen.generate_raw_u32(out)
    if distribution == "raw64":
        return gen.generate_raw_u64(out)
    if distribution == "uniform_f32":
        return gen.generate_uniform_f32(out)
    if distribution == "uniform_f64":
        return gen.generate_uniform_f64(out)
    if distribution == "normal_f32":
        return gen.generate_normal_f32(out, mean=mean, stddev=stddev)
    if distribution == "normal_f64":
        return gen.generate_normal_f64(out, mean=mean, stddev=stddev)
    if distribution == "lognormal_f32":
        return gen.generate_lognormal_f32(out, mean=mean, stddev=stddev)
    if distribution == "lognormal_f64":
        return gen.generate_lognormal_f64(out, mean=mean, stddev=stddev)
    if distribution == "poisson_u32":
        return gen.generate_poisson_u32(out, lambda_val=lambda_val)
    raise ValueError(f"Unsupported cuRAND distribution: {distribution}")
