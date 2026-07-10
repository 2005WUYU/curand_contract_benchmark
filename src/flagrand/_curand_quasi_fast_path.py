from __future__ import annotations

import torch

from flagrand._curand_handle import (
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
    CURAND_RNG_QUASI_SOBOL32,
    CURAND_RNG_QUASI_SOBOL64,
    CurandGenerator,
)


_QUASI32_TYPES = {
    CURAND_RNG_QUASI_SOBOL32,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
}
_QUASI64_TYPES = {
    CURAND_RNG_QUASI_SOBOL64,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
}
_QUASI_TYPES = _QUASI32_TYPES | _QUASI64_TYPES


def try_generate_quasi_raw(
    generator,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    rng_type, engine, block_size, num_warps = prepared
    if rng_type in _QUASI32_TYPES and out.dtype == torch.int32:
        engine.generate(out, block_size=block_size, num_warps=num_warps)
        return True
    if rng_type in _QUASI64_TYPES and out.dtype == torch.int64:
        engine.generate_long_long(out, block_size=block_size, num_warps=num_warps)
        return True
    return False


def try_generate_quasi_uniform(
    generator,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    _, engine, block_size, num_warps = prepared
    engine.generate_uniform(out, block_size=block_size, num_warps=num_warps)
    return True


def try_generate_quasi_normal(
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
    _, engine, block_size, num_warps = prepared
    engine.generate_normal(
        out,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


def try_generate_quasi_lognormal(
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
    _, engine, block_size, num_warps = prepared
    engine.generate_lognormal(
        out,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


def _prepare(generator, out, kwargs):
    if not isinstance(generator, CurandGenerator) or generator.rng_type not in _QUASI_TYPES:
        return None
    config = _launch_config(kwargs)
    if config is None or out.device.type != "cuda" or not out.is_contiguous():
        return None
    return generator.rng_type, generator.engine, config[0], config[1]


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
