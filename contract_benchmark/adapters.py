from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from contract_benchmark import curand_ctypes as curand


@dataclass(frozen=True)
class GeneratorInfo:
    name: str
    curand_name: str
    flagrand_class_name: str
    kind: str
    raw_dtype: torch.dtype
    supports_curand_host: bool
    supports_flagrand: bool
    supports_raw32: bool
    supports_raw64: bool
    supports_distributions_f32: bool
    supports_seed: bool
    supports_offset: bool
    notes: list[str]


GENERATOR_INFOS: dict[str, GeneratorInfo] = {
    "philox4x32_10": GeneratorInfo(
        "philox4x32_10",
        "philox4x32_10",
        "PhiloxGenerator",
        "prng",
        torch.int32,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        ["FlagRand Philox currently uses a benchmark-visible counter mapping, not cuRAND Host ordering."],
    ),
    "xorwow": GeneratorInfo(
        "xorwow",
        "xorwow",
        "XorwowGenerator",
        "prng",
        torch.int32,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        [],
    ),
    "mrg32k3a": GeneratorInfo(
        "mrg32k3a",
        "mrg32k3a",
        "Mrg32k3aGenerator",
        "prng",
        torch.int32,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        [],
    ),
    "mtgp32": GeneratorInfo(
        "mtgp32",
        "mtgp32",
        "Mtgp32Generator",
        "prng",
        torch.int32,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        ["FlagRand MTGP32 rejects non-zero offset."],
    ),
    "mt19937": GeneratorInfo(
        "mt19937",
        "mt19937",
        "Mt19937Generator",
        "prng",
        torch.int32,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        ["cuRAND MT19937 is Host API only; FlagRand implementation has its own state mapping."],
    ),
    "sobol32": GeneratorInfo(
        "sobol32",
        "sobol32",
        "Sobol32Generator",
        "qrng",
        torch.int32,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        ["Sobol has dimensions/offset semantics, not seed semantics."],
    ),
    "scrambled_sobol32": GeneratorInfo(
        name="scrambled_sobol32",
        curand_name="scrambled_sobol32",
        flagrand_class_name="ScrambledSobol32Generator",
        kind="qrng",
        raw_dtype=torch.int32,
        supports_curand_host=True,
        supports_flagrand=True,
        supports_raw32=True,
        supports_raw64=False,
        supports_distributions_f32=True,
        supports_seed=False,
        supports_offset=True,
        notes=["Scrambled Sobol seed is scramble-related, not PRNG sequence seed."],
    ),
    "sobol64": GeneratorInfo(
        "sobol64",
        "sobol64",
        "Sobol64Generator",
        "qrng",
        torch.int64,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        ["cuRAND Host raw64 is native only for Sobol64 families."],
    ),
    "scrambled_sobol64": GeneratorInfo(
        name="scrambled_sobol64",
        curand_name="scrambled_sobol64",
        flagrand_class_name="ScrambledSobol64Generator",
        kind="qrng",
        raw_dtype=torch.int64,
        supports_curand_host=True,
        supports_flagrand=True,
        supports_raw32=False,
        supports_raw64=True,
        supports_distributions_f32=True,
        supports_seed=False,
        supports_offset=True,
        notes=["cuRAND Host raw64 is native only for Sobol64 families."],
    ),
}


def make_curand_generator(
    generator: str,
    *,
    seed: int,
    offset: int = 0,
    ordering: str | None = "legacy",
    dimensions: int | None = None,
) -> curand.CurandGenerator:
    return curand.CurandGenerator(
        generator,
        seed=seed,
        offset=offset,
        ordering=ordering,
        dimensions=dimensions,
    )


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


def curand_generate_by_distribution(
    gen: curand.CurandGenerator,
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


def capability_matrix() -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "curand_host": curand.library_load_report(),
        "generators": {},
        "device_api_extension": optional_device_extension_status(),
        "curanddx": {
            "available": False,
            "unsupported_reason": "cuRANDDx headers/build integration are not configured in this local repository.",
        },
    }
    for name, info in GENERATOR_INFOS.items():
        matrix["generators"][name] = {
            "kind": info.kind,
            "curand_host": info.supports_curand_host,
            "flagrand": info.supports_flagrand,
            "raw32": info.supports_raw32,
            "raw64_native": info.supports_raw64,
            "distributions_f32": info.supports_distributions_f32,
            "supports_seed": info.supports_seed,
            "supports_offset": info.supports_offset,
            "notes": info.notes,
        }
    return matrix


def optional_device_extension_status() -> dict[str, Any]:
    try:
        from contract_benchmark.optional_device_api import find_built_curand_device_extension
    except Exception as exc:
        return {"available": False, "unsupported_reason": f"optional loader import failed: {exc}"}
    module, reason = find_built_curand_device_extension()
    return {
        "available": module is not None,
        "unsupported_reason": reason if module is None else None,
    }


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


Operation = Callable[[], object]
