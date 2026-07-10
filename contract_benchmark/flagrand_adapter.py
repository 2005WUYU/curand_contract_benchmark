from __future__ import annotations

from typing import Any

import torch

from contract_benchmark.generator_registry import GENERATOR_INFOS


def make_flagrand_generator(
    generator: str,
    *,
    seed: int,
    offset: int = 0,
    device: torch.device | str = "cuda",
    dimensions: int | None = None,
):
    if generator not in GENERATOR_INFOS:
        raise ValueError(f"Unsupported FlagRand generator: {generator}")
    del device  # The refactored facade binds the device when an output tensor is supplied.
    curand = _flagrand_modules()["curand"]
    rng_type = _flagrand_rng_type(generator, curand)
    return curand.create_generator(
        rng_type,
        seed=int(seed),
        offset=int(offset),
        dimensions=int(dimensions or 1),
    )


def flagrand_generate_raw(out: torch.Tensor, generator_obj: object) -> torch.Tensor:
    curand = _flagrand_modules()["curand"]
    if out.dtype == torch.int32:
        return curand.generate(generator_obj, out)
    if out.dtype == torch.int64:
        return curand.generate_long_long(generator_obj, out)
    raise TypeError(f"FlagRand raw output must use torch.int32 or torch.int64, got {out.dtype}.")


def flagrand_generate_uniform(out: torch.Tensor, generator_obj: object) -> torch.Tensor:
    curand = _flagrand_modules()["curand"]
    if out.dtype == torch.float32:
        return curand.generate_uniform(generator_obj, out)
    if out.dtype == torch.float64:
        return curand.generate_uniform_double(generator_obj, out)
    raise TypeError(f"FlagRand uniform output must use torch.float32 or torch.float64, got {out.dtype}.")


def flagrand_generate_normal(
    out: torch.Tensor,
    generator_obj: object,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
) -> torch.Tensor:
    curand = _flagrand_modules()["curand"]
    if out.dtype == torch.float32:
        return curand.generate_normal(generator_obj, out, mean=mean, stddev=stddev)
    if out.dtype == torch.float64:
        return curand.generate_normal_double(generator_obj, out, mean=mean, stddev=stddev)
    raise TypeError(f"FlagRand normal output must use torch.float32 or torch.float64, got {out.dtype}.")


def flagrand_generate_lognormal(
    out: torch.Tensor,
    generator_obj: object,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
) -> torch.Tensor:
    curand = _flagrand_modules()["curand"]
    if out.dtype == torch.float32:
        return curand.generate_lognormal(generator_obj, out, mean=mean, stddev=stddev)
    if out.dtype == torch.float64:
        return curand.generate_lognormal_double(generator_obj, out, mean=mean, stddev=stddev)
    raise TypeError(f"FlagRand lognormal output must use torch.float32 or torch.float64, got {out.dtype}.")


def flagrand_generate_poisson(
    out: torch.Tensor,
    generator_obj: object,
    *,
    lambda_val: float,
) -> torch.Tensor:
    return _flagrand_modules()["curand"].generate_poisson(
        generator_obj,
        out,
        lambda_val=lambda_val,
    )


def flagrand_generate_by_distribution(
    generator_obj: object,
    out: torch.Tensor,
    distribution: str,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
    lambda_val: float = 10.0,
) -> torch.Tensor:
    if distribution in ("raw32", "raw64"):
        return flagrand_generate_raw(out, generator_obj)
    if distribution in {"uniform_f32", "uniform_f64"}:
        return flagrand_generate_uniform(out, generator_obj)
    if distribution in {"normal_f32", "normal_f64"}:
        return flagrand_generate_normal(out, generator_obj, mean=mean, stddev=stddev)
    if distribution in {"lognormal_f32", "lognormal_f64"}:
        return flagrand_generate_lognormal(out, generator_obj, mean=mean, stddev=stddev)
    if distribution == "poisson_u32":
        return flagrand_generate_poisson(out, generator_obj, lambda_val=lambda_val)
    raise ValueError(f"Unsupported FlagRand distribution: {distribution}")


_FLAGRAND_CACHE: dict[str, Any] | None = None


def _flagrand_modules() -> dict[str, Any]:
    global _FLAGRAND_CACHE
    if _FLAGRAND_CACHE is not None:
        return _FLAGRAND_CACHE
    from flagrand import curand

    _FLAGRAND_CACHE = {"curand": curand}
    return _FLAGRAND_CACHE


def _flagrand_rng_type(generator: str, curand: Any) -> str:
    mapping = {
        "philox4x32_10": curand.CURAND_RNG_PSEUDO_PHILOX4_32_10,
        "xorwow": curand.CURAND_RNG_PSEUDO_XORWOW,
        "mrg32k3a": curand.CURAND_RNG_PSEUDO_MRG32K3A,
        "mtgp32": curand.CURAND_RNG_PSEUDO_MTGP32,
        "mt19937": curand.CURAND_RNG_PSEUDO_MT19937,
        "sobol32": curand.CURAND_RNG_QUASI_SOBOL32,
        "scrambled_sobol32": curand.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
        "sobol64": curand.CURAND_RNG_QUASI_SOBOL64,
        "scrambled_sobol64": curand.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
    }
    return mapping[generator]
