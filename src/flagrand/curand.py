from __future__ import annotations

from typing import Any

import torch

from flagrand._curand_fast_path import (
    is_philox_generator,
    try_generate_lognormal,
    try_generate_normal,
    try_generate_poisson,
    try_generate_raw,
    try_generate_uniform,
)
from flagrand._curand_quasi_fast_path import (
    try_generate_quasi_lognormal,
    try_generate_quasi_normal,
    try_generate_quasi_raw,
    try_generate_quasi_uniform,
)
from flagrand._curand_stateful_fast_path import (
    try_generate_stateful_lognormal,
    try_generate_stateful_normal,
    try_generate_stateful_poisson,
    try_generate_stateful_raw,
    try_generate_stateful_uniform,
)
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
def generate(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.int32, "generate")
    if is_philox_generator(generator):
        handled = try_generate_raw(generator, out, kwargs)
    else:
        handled = try_generate_stateful_raw(generator, out, kwargs) or try_generate_raw(
            generator, out, kwargs
        )
    if handled or try_generate_quasi_raw(generator, out, kwargs):
        return out
    return _generate_raw(out, _engine(generator), **kwargs)


def generate_long_long(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.int64, "generate_long_long")
    if try_generate_raw(generator, out, kwargs) or try_generate_quasi_raw(generator, out, kwargs):
        return out
    return _generate_raw(out, _engine(generator), **kwargs)


def generate_uniform(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_uniform")
    if is_philox_generator(generator):
        handled = try_generate_uniform(generator, out, kwargs)
    else:
        handled = try_generate_stateful_uniform(
            generator, out, kwargs
        ) or try_generate_uniform(generator, out, kwargs)
    if handled or try_generate_quasi_uniform(generator, out, kwargs):
        return out
    return _generate_uniform(out, _engine(generator), **kwargs)


def generate_uniform_double(
    generator: CurandGenerator | Any,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_uniform_double")
    if try_generate_uniform(generator, out, kwargs) or try_generate_quasi_uniform(generator, out, kwargs):
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
    if try_generate_stateful_normal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ) or try_generate_normal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ) or try_generate_quasi_normal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ):
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
    if try_generate_normal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ) or try_generate_quasi_normal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ):
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
    if try_generate_stateful_lognormal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ) or try_generate_lognormal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ) or try_generate_quasi_lognormal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ):
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
    if try_generate_lognormal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ) or try_generate_quasi_lognormal(
        generator, out, mean=mean, stddev=stddev, kwargs=kwargs
    ):
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
    if try_generate_stateful_poisson(
        generator, out, lambda_val=lambda_val, kwargs=kwargs
    ) or try_generate_poisson(
        generator, out, lambda_val=lambda_val, kwargs=kwargs
    ):
        return out
    return _generate_poisson(out, _engine(generator), lambda_val=lambda_val, **kwargs)


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
