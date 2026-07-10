from __future__ import annotations

from dataclasses import dataclass

from flagrand.rng import (
    Mrg32k3aGenerator,
    Mt19937Generator,
    Mtgp32Generator,
    PhiloxGenerator,
    ScrambledSobol32Generator,
    ScrambledSobol64Generator,
    Sobol32Generator,
    Sobol64Generator,
    XorwowGenerator,
)

CURAND_RNG_PSEUDO_XORWOW = "pseudo_xorwow"
CURAND_RNG_PSEUDO_MRG32K3A = "pseudo_mrg32k3a"
CURAND_RNG_PSEUDO_MTGP32 = "pseudo_mtgp32"
CURAND_RNG_PSEUDO_MT19937 = "pseudo_mt19937"
CURAND_RNG_PSEUDO_PHILOX4_32_10 = "pseudo_philox4_32_10"
CURAND_RNG_QUASI_SOBOL32 = "quasi_sobol32"
CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = "quasi_scrambled_sobol32"
CURAND_RNG_QUASI_SOBOL64 = "quasi_sobol64"
CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = "quasi_scrambled_sobol64"

PSEUDO_TYPES = {
    CURAND_RNG_PSEUDO_XORWOW: XorwowGenerator,
    CURAND_RNG_PSEUDO_MRG32K3A: Mrg32k3aGenerator,
    CURAND_RNG_PSEUDO_MTGP32: Mtgp32Generator,
    CURAND_RNG_PSEUDO_MT19937: Mt19937Generator,
    CURAND_RNG_PSEUDO_PHILOX4_32_10: PhiloxGenerator,
}
QUASI_TYPES = {
    CURAND_RNG_QUASI_SOBOL32: Sobol32Generator,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32: ScrambledSobol32Generator,
    CURAND_RNG_QUASI_SOBOL64: Sobol64Generator,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: ScrambledSobol64Generator,
}

_ALIASES = {
    "xorwow": CURAND_RNG_PSEUDO_XORWOW,
    "mrg32k3a": CURAND_RNG_PSEUDO_MRG32K3A,
    "mtgp32": CURAND_RNG_PSEUDO_MTGP32,
    "mt19937": CURAND_RNG_PSEUDO_MT19937,
    "philox": CURAND_RNG_PSEUDO_PHILOX4_32_10,
    "philox4_32_10": CURAND_RNG_PSEUDO_PHILOX4_32_10,
    "sobol32": CURAND_RNG_QUASI_SOBOL32,
    "scrambled_sobol32": CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
    "sobol64": CURAND_RNG_QUASI_SOBOL64,
    "scrambled_sobol64": CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
}


@dataclass
class CurandGenerator:
    rng_type: str
    engine: object

    @property
    def seed(self) -> int | None:
        return getattr(self.engine, "seed", None)

    @property
    def offset(self) -> int:
        return int(getattr(self.engine, "offset", 0))

    @property
    def dimensions(self) -> int | None:
        return getattr(self.engine, "dimensions", None)


def create_generator(
    rng_type: str = CURAND_RNG_PSEUDO_PHILOX4_32_10,
    *,
    seed: int = 0,
    offset: int = 0,
    dimensions: int = 1,
) -> CurandGenerator:
    resolved = resolve_rng_type(rng_type)
    if resolved in PSEUDO_TYPES:
        engine = PSEUDO_TYPES[resolved](seed=int(seed), offset=int(offset))
    elif resolved in QUASI_TYPES:
        engine = QUASI_TYPES[resolved](dimensions=int(dimensions), offset=int(offset))
    else:
        raise ValueError(f"Unsupported cuRAND rng_type: {rng_type}")
    return CurandGenerator(rng_type=resolved, engine=engine)


def set_pseudo_random_generator_seed(generator: CurandGenerator, seed: int) -> CurandGenerator:
    if generator.rng_type not in PSEUDO_TYPES:
        raise ValueError(f"{generator.rng_type} does not support pseudo-random seeds.")
    generator.engine.seed = int(seed)
    generator.engine.offset = 0
    return generator


def set_generator_offset(generator: CurandGenerator, offset: int) -> CurandGenerator:
    if generator.rng_type in QUASI_TYPES:
        generator.engine = QUASI_TYPES[generator.rng_type](
            dimensions=int(generator.dimensions or 1),
            offset=int(offset),
        )
    else:
        generator.engine.offset = int(offset)
    return generator


def set_quasi_random_generator_dimensions(
    generator: CurandGenerator,
    dimensions: int,
) -> CurandGenerator:
    if generator.rng_type not in QUASI_TYPES:
        raise ValueError(f"{generator.rng_type} does not support quasi-random dimensions.")
    generator.engine = QUASI_TYPES[generator.rng_type](
        dimensions=int(dimensions),
        offset=generator.offset,
    )
    return generator


def resolve_rng_type(rng_type: str) -> str:
    return _ALIASES.get(rng_type, rng_type)
