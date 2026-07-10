from __future__ import annotations

import torch

from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_MRG32K3A,
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CURAND_RNG_PSEUDO_XORWOW,
    CurandGenerator,
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
from flagrand.fused._internal.state_prng_direct import (
    generate_mrg32k3a_lognormal_f32,
    generate_mrg32k3a_normal_f32,
    generate_mrg32k3a_poisson_u32,
    generate_mrg32k3a_uniform_f32,
    generate_xorwow_lognormal_f32,
    generate_xorwow_normal_f32,
    generate_xorwow_poisson_u32,
    generate_xorwow_uniform_f32,
)
from flagrand.fused.poisson import _small_poisson_max_k
from flagrand.rng.philox import generate_philox_raw_u32


_STATE_UNIFORM = {
    CURAND_RNG_PSEUDO_XORWOW: generate_xorwow_uniform_f32,
    CURAND_RNG_PSEUDO_MRG32K3A: generate_mrg32k3a_uniform_f32,
}
_STATE_NORMAL = {
    CURAND_RNG_PSEUDO_XORWOW: generate_xorwow_normal_f32,
    CURAND_RNG_PSEUDO_MRG32K3A: generate_mrg32k3a_normal_f32,
}
_STATE_LOGNORMAL = {
    CURAND_RNG_PSEUDO_XORWOW: generate_xorwow_lognormal_f32,
    CURAND_RNG_PSEUDO_MRG32K3A: generate_mrg32k3a_lognormal_f32,
}
_STATE_POISSON = {
    CURAND_RNG_PSEUDO_XORWOW: generate_xorwow_poisson_u32,
    CURAND_RNG_PSEUDO_MRG32K3A: generate_mrg32k3a_poisson_u32,
}
_DIRECT_TYPES = {
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CURAND_RNG_PSEUDO_XORWOW,
    CURAND_RNG_PSEUDO_MRG32K3A,
}


def try_generate_raw(generator, out: torch.Tensor, kwargs: dict[str, object]) -> bool:
    direct = _direct_engine(generator)
    if direct is None or direct[0] not in _DIRECT_TYPES:
        return False
    config = _launch_config(kwargs)
    if config is None or not _supports_direct_output(out):
        return False
    if direct[0] == CURAND_RNG_PSEUDO_PHILOX4_32_10 and out.dtype == torch.int32:
        generate_philox_raw_u32(
            out,
            direct[1],
            block_size=config[0],
            num_warps=config[1],
        )
        return True
    if direct[0] in _STATE_UNIFORM and out.dtype == torch.int32:
        direct[1].generate(out)
        return True
    return False


def try_generate_uniform(generator, out: torch.Tensor, kwargs: dict[str, object]) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    rng_type, engine, block_size, num_warps = prepared
    if out.numel() == 0:
        return True
    if rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        operation = generate_philox_uniform_f64 if out.dtype == torch.float64 else generate_philox_uniform_f32
        operation(out, engine, block_size=block_size, num_warps=num_warps)
        return True
    operation = _STATE_UNIFORM.get(rng_type) if out.dtype == torch.float32 else None
    if operation is None:
        return False
    operation(out, engine)
    return True


def try_generate_normal(
    generator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    rng_type, engine, block_size, num_warps = prepared
    if out.numel() == 0:
        return True
    if rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        operation = generate_philox_normal_f64 if out.dtype == torch.float64 else generate_philox_normal_f32
        operation(out, engine, mean=mean, stddev=stddev, block_size=block_size, num_warps=num_warps)
        return True
    operation = _STATE_NORMAL.get(rng_type) if out.dtype == torch.float32 else None
    if operation is None:
        return False
    operation(out, engine, mean=mean, stddev=stddev)
    return True


def try_generate_lognormal(
    generator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    rng_type, engine, block_size, num_warps = prepared
    if out.numel() == 0:
        return True
    if rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        operation = generate_philox_lognormal_f64 if out.dtype == torch.float64 else generate_philox_lognormal_f32
        operation(out, engine, mean=mean, stddev=stddev, block_size=block_size, num_warps=num_warps)
        return True
    operation = _STATE_LOGNORMAL.get(rng_type) if out.dtype == torch.float32 else None
    if operation is None:
        return False
    operation(out, engine, mean=mean, stddev=stddev)
    return True


def try_generate_poisson(
    generator,
    out: torch.Tensor,
    *,
    lambda_val: float,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    rng_type, engine, block_size, num_warps = prepared
    n = out.numel()
    if n == 0:
        return True
    if lambda_val <= 0:
        raise ValueError(f"generate_poisson: lambda must be > 0, got {lambda_val}.")
    if lambda_val >= 30.0 and n % 2:
        raise ValueError(
            f"generate_poisson: lambda >= 30 currently requires an even element count, got {n}."
        )
    if rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        generate_philox_poisson_u32(
            out,
            engine,
            lambda_val=lambda_val,
            block_size=block_size,
            num_warps=num_warps,
        )
        return True
    operation = _STATE_POISSON.get(rng_type)
    if operation is None:
        return False
    max_k = _small_poisson_max_k(lambda_val) if lambda_val < 30.0 else 0
    operation(out, engine, lambda_val=lambda_val, max_k=max_k)
    return True


def _prepare(generator, out, kwargs):
    direct = _direct_engine(generator)
    if direct is None or direct[0] not in _DIRECT_TYPES:
        return None
    config = _launch_config(kwargs)
    if config is None or not _supports_direct_output(out):
        return None
    return direct[0], direct[1], config[0], config[1]


def _direct_engine(generator) -> tuple[str, object] | None:
    if not isinstance(generator, CurandGenerator):
        return None
    return generator.rng_type, generator.engine


def _launch_config(kwargs: dict[str, object]) -> tuple[int, int] | None:
    if not kwargs:
        return 512, 4
    if any(name not in {"block_size", "num_warps"} for name in kwargs):
        return None
    block_size = kwargs.get("block_size", 512)
    num_warps = kwargs.get("num_warps", 4)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    return block_size, num_warps


def _supports_direct_output(out: torch.Tensor) -> bool:
    return out.device.type == "cuda" and out.is_contiguous()
