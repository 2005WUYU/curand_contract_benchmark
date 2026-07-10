from __future__ import annotations

from typing import Any

import torch

from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_MRG32K3A,
    CURAND_RNG_PSEUDO_MT19937,
    CURAND_RNG_PSEUDO_MTGP32,
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CURAND_RNG_PSEUDO_XORWOW,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
    CURAND_RNG_QUASI_SOBOL32,
    CURAND_RNG_QUASI_SOBOL64,
    CurandGenerator,
    create_generator,
    set_generator_offset,
    set_pseudo_random_generator_seed,
    set_quasi_random_generator_dimensions,
)
from flagrand.fused import (
    generate_lognormal as _generate_lognormal,
    generate_normal as _generate_normal,
    generate_poisson as _generate_poisson,
    generate_raw as _generate_raw,
    generate_uniform as _generate_uniform,
)
from flagrand.fused._internal.philox_direct import (
    generate_philox_lognormal_f32,
    generate_philox_normal_f32,
    generate_philox_poisson_u32,
    generate_philox_uniform_f32,
)
from flagrand.fused._internal.philox_direct_f64 import (
    generate_philox_lognormal_f64,
    generate_philox_normal_f64,
    generate_philox_uniform_f64,
)
from flagrand.rng.philox import generate_philox_raw_u32


def generate(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.int32, "generate")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            return generate_philox_raw_u32(
                out,
                engine,
                block_size=config[0],
                num_warps=config[1],
            )
    return _generate_raw(out, _engine(generator), **kwargs)


def generate_long_long(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.int64, "generate_long_long")
    return _generate_raw(out, _engine(generator), **kwargs)


def generate_uniform(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_uniform")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_uniform_f32(
                out,
                engine,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_uniform(out, _engine(generator), **kwargs)


def generate_uniform_double(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_uniform_double")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_uniform_f64(
                out,
                engine,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_uniform(out, _engine(generator), **kwargs)


def generate_normal(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_normal")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_normal_f32(
                out,
                engine,
                mean=mean,
                stddev=stddev,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_normal(out, _engine(generator), mean=mean, stddev=stddev, **kwargs)


def generate_normal_double(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_normal_double")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_normal_f64(
                out,
                engine,
                mean=mean,
                stddev=stddev,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_normal(out, _engine(generator), mean=mean, stddev=stddev, **kwargs)


def generate_lognormal(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_lognormal")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_lognormal_f32(
                out,
                engine,
                mean=mean,
                stddev=stddev,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_lognormal(out, _engine(generator), mean=mean, stddev=stddev, **kwargs)


def generate_lognormal_double(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_lognormal_double")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_lognormal_f64(
                out,
                engine,
                mean=mean,
                stddev=stddev,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_lognormal(out, _engine(generator), mean=mean, stddev=stddev, **kwargs)


def generate_poisson(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    *,
    lambda_val: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.int32, "generate_poisson")
    engine = _fast_philox_engine(generator)
    if engine is not None:
        config = _fast_launch_config(kwargs)
        if config is not None and _supports_fast_output(out):
            if out.numel() == 0:
                return out
            generate_philox_poisson_u32(
                out,
                engine,
                lambda_val=lambda_val,
                block_size=config[0],
                num_warps=config[1],
            )
            return out
    return _generate_poisson(out, _engine(generator), lambda_val=lambda_val, **kwargs)


def _fast_philox_engine(generator: CurandGenerator | Any) -> object | None:
    if (
        isinstance(generator, CurandGenerator)
        and generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10
    ):
        return generator.engine
    return None


def _fast_launch_config(kwargs: dict[str, object]) -> tuple[int, int] | None:
    if any(name not in {"block_size", "num_warps"} for name in kwargs):
        return None
    block_size = kwargs.get("block_size", 512)
    num_warps = kwargs.get("num_warps", 4)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    return block_size, num_warps


def _supports_fast_output(out: torch.Tensor) -> bool:
    return out.device.type == "cuda" and out.is_contiguous()


def _engine(generator: CurandGenerator | Any) -> object:
    return generator.engine if isinstance(generator, CurandGenerator) else generator


def _require_dtype(out: torch.Tensor, dtype: torch.dtype, op_name: str) -> None:
    if out.dtype != dtype:
        raise TypeError(f"{op_name}: expected output dtype {dtype}, got {out.dtype}.")


__all__ = [
    "CURAND_RNG_PSEUDO_XORWOW",
    "CURAND_RNG_PSEUDO_MRG32K3A",
    "CURAND_RNG_PSEUDO_MTGP32",
    "CURAND_RNG_PSEUDO_MT19937",
    "CURAND_RNG_PSEUDO_PHILOX4_32_10",
    "CURAND_RNG_QUASI_SOBOL32",
    "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",
    "CURAND_RNG_QUASI_SOBOL64",
    "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",
    "CurandGenerator",
    "create_generator",
    "set_pseudo_random_generator_seed",
    "set_generator_offset",
    "set_quasi_random_generator_dimensions",
    "generate",
    "generate_long_long",
    "generate_uniform",
    "generate_uniform_double",
    "generate_normal",
    "generate_normal_double",
    "generate_lognormal",
    "generate_lognormal_double",
    "generate_poisson",
]
