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
    modules = _flagrand_modules()
    classes = modules["classes"]
    if generator not in GENERATOR_INFOS:
        raise ValueError(f"Unsupported FlagRand generator: {generator}")
    cls = classes[GENERATOR_INFOS[generator].flagrand_class_name]
    kwargs: dict[str, Any] = {"offset": int(offset)}
    if GENERATOR_INFOS[generator].supports_seed:
        kwargs["seed"] = int(seed)
    if GENERATOR_INFOS[generator].kind == "qrng":
        kwargs["dimensions"] = int(dimensions or 1)
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("dimensions", None)
        kwargs["device"] = str(device)
        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return cls(**kwargs)


def flagrand_generate_raw(out: torch.Tensor, generator_obj: object) -> torch.Tensor:
    return _flagrand_modules()["generate_raw"](out, generator_obj)


def flagrand_generate_uniform(out: torch.Tensor, generator_obj: object) -> torch.Tensor:
    return _flagrand_modules()["generate_uniform"](out, generator_obj)


def flagrand_generate_normal(
    out: torch.Tensor,
    generator_obj: object,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
) -> torch.Tensor:
    return _flagrand_modules()["generate_normal"](out, generator_obj, mean=mean, stddev=stddev)


def flagrand_generate_lognormal(
    out: torch.Tensor,
    generator_obj: object,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
) -> torch.Tensor:
    return _flagrand_modules()["generate_lognormal"](out, generator_obj, mean=mean, stddev=stddev)


def flagrand_generate_poisson(
    out: torch.Tensor,
    generator_obj: object,
    *,
    lambda_val: float,
) -> torch.Tensor:
    return _flagrand_modules()["generate_poisson"](out, generator_obj, lambda_val=lambda_val)


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
    if distribution == "uniform_f32":
        return flagrand_generate_uniform(out, generator_obj)
    if distribution == "normal_f32":
        return flagrand_generate_normal(out, generator_obj, mean=mean, stddev=stddev)
    if distribution == "lognormal_f32":
        return flagrand_generate_lognormal(out, generator_obj, mean=mean, stddev=stddev)
    if distribution == "poisson_u32":
        return flagrand_generate_poisson(out, generator_obj, lambda_val=lambda_val)
    raise ValueError(f"Unsupported FlagRand distribution: {distribution}")


_FLAGRAND_CACHE: dict[str, Any] | None = None


def _flagrand_modules() -> dict[str, Any]:
    global _FLAGRAND_CACHE
    if _FLAGRAND_CACHE is not None:
        return _FLAGRAND_CACHE
    from flagrand.fused.lognormal import generate_lognormal
    from flagrand.fused.normal import generate_normal
    from flagrand.fused.poisson import generate_poisson
    from flagrand.fused.raw import generate_raw
    from flagrand.fused.uniform import generate_uniform
    from flagrand.rng.mrg32k3a import Mrg32k3aGenerator
    from flagrand.rng.mt19937 import Mt19937Generator
    from flagrand.rng.mtgp32 import Mtgp32Generator
    from flagrand.rng.philox import PhiloxGenerator
    from flagrand.rng.scrambled_sobol32 import ScrambledSobol32Generator
    from flagrand.rng.scrambled_sobol64 import ScrambledSobol64Generator
    from flagrand.rng.sobol32 import Sobol32Generator
    from flagrand.rng.sobol64 import Sobol64Generator
    from flagrand.rng.xorwow import XorwowGenerator

    _FLAGRAND_CACHE = {
        "generate_raw": generate_raw,
        "generate_uniform": generate_uniform,
        "generate_normal": generate_normal,
        "generate_lognormal": generate_lognormal,
        "generate_poisson": generate_poisson,
        "classes": {
            "PhiloxGenerator": PhiloxGenerator,
            "XorwowGenerator": XorwowGenerator,
            "Mrg32k3aGenerator": Mrg32k3aGenerator,
            "Mtgp32Generator": Mtgp32Generator,
            "Mt19937Generator": Mt19937Generator,
            "Sobol32Generator": Sobol32Generator,
            "ScrambledSobol32Generator": ScrambledSobol32Generator,
            "Sobol64Generator": Sobol64Generator,
            "ScrambledSobol64Generator": ScrambledSobol64Generator,
        },
    }
    return _FLAGRAND_CACHE
