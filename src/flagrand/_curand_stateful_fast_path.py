from __future__ import annotations

import torch

from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_MT19937,
    CURAND_RNG_PSEUDO_MTGP32,
    CurandGenerator,
)
from flagrand.rng._stateful_output import (
    RAW_OUTPUT,
    UNIFORM_OUTPUT,
    normal_output,
    poisson_output_for_lambda,
)


_STATEFUL_TYPES = {CURAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_MT19937}


def try_generate_stateful_raw(
    generator,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, num_warps = prepared
    if out.numel():
        engine._generate_prepared(out, RAW_OUTPUT, num_warps=num_warps)
    return True


def try_generate_stateful_uniform(
    generator,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, num_warps = prepared
    if out.numel():
        engine._generate_prepared(out, UNIFORM_OUTPUT, num_warps=num_warps)
    return True


def try_generate_stateful_normal(
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
    engine, num_warps = prepared
    if out.numel() == 0:
        return True
    _require_even_pairs(out, "generate_normal")
    engine._generate_prepared(
        out,
        normal_output(mean, stddev, lognormal=False),
        num_warps=num_warps,
    )
    return True


def try_generate_stateful_lognormal(
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
    engine, num_warps = prepared
    if out.numel() == 0:
        return True
    _require_even_pairs(out, "generate_lognormal")
    engine._generate_prepared(
        out,
        normal_output(mean, stddev, lognormal=True),
        num_warps=num_warps,
    )
    return True


def try_generate_stateful_poisson(
    generator,
    out: torch.Tensor,
    *,
    lambda_val: float,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, num_warps = prepared
    if out.numel() == 0:
        return True
    if lambda_val <= 0:
        raise ValueError(f"generate_poisson: lambda must be > 0, got {lambda_val}.")
    if lambda_val >= 30.0 and out.numel() % 2:
        raise ValueError(
            f"generate_poisson: lambda >= 30 currently requires an even element count, got {out.numel()}."
        )
    engine._generate_prepared(
        out,
        poisson_output_for_lambda(lambda_val),
        num_warps=num_warps,
    )
    return True


def _prepare(generator, out, kwargs):
    if not isinstance(generator, CurandGenerator) or generator.rng_type not in _STATEFUL_TYPES:
        return None
    if any(name not in {"block_size", "num_warps"} for name in kwargs):
        return None
    if out.device.type != "cuda" or not out.is_contiguous():
        return None
    default_block_size = 256 if generator.rng_type == CURAND_RNG_PSEUDO_MTGP32 else 624
    default_num_warps = 8 if generator.rng_type == CURAND_RNG_PSEUDO_MTGP32 else 4
    if not kwargs:
        return generator.engine, default_num_warps
    block_size = kwargs.get("block_size", default_block_size)
    num_warps = kwargs.get("num_warps", default_num_warps)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if generator.rng_type == CURAND_RNG_PSEUDO_MTGP32 and block_size != 256:
        raise ValueError("MTGP32 uses a fixed block_size=256 to preserve state ordering.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    return generator.engine, num_warps


def _require_even_pairs(out: torch.Tensor, op_name: str) -> None:
    if out.numel() % 2:
        raise ValueError(
            f"{op_name}: Box-Muller output requires an even element count, got {out.numel()}."
        )
