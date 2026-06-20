from __future__ import annotations

from dataclasses import dataclass

import torch


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
