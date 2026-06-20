from __future__ import annotations

import torch

from flagrand.rng.mrg32k3a import Mrg32k3aGenerator
from flagrand.rng.mt19937 import Mt19937Generator
from flagrand.rng.mtgp32 import Mtgp32Generator
from flagrand.rng.philox import PhiloxGenerator
from flagrand.rng.scrambled_sobol32 import ScrambledSobol32Generator
from flagrand.rng.scrambled_sobol64 import ScrambledSobol64Generator
from flagrand.rng.sobol32 import Sobol32Generator
from flagrand.rng.sobol64 import Sobol64Generator
from flagrand.rng.xorwow import XorwowGenerator

GENERATOR_PHILOX = 0
GENERATOR_XORWOW = 1
GENERATOR_MRG32K3A = 2
GENERATOR_MT19937 = 3
GENERATOR_MTGP32 = 4
GENERATOR_SOBOL32 = 5
GENERATOR_SOBOL64 = 6
GENERATOR_SCRAMBLED_SOBOL32 = 7
GENERATOR_SCRAMBLED_SOBOL64 = 8


def get_generator_type(generator) -> int:

    if isinstance(generator, PhiloxGenerator):
        return GENERATOR_PHILOX
    elif isinstance(generator, XorwowGenerator):
        return GENERATOR_XORWOW
    elif isinstance(generator, Mrg32k3aGenerator):
        return GENERATOR_MRG32K3A
    elif isinstance(generator, Mt19937Generator):
        return GENERATOR_MT19937
    elif isinstance(generator, Mtgp32Generator):
        return GENERATOR_MTGP32
    elif isinstance(generator, Sobol32Generator):
        return GENERATOR_SOBOL32
    elif isinstance(generator, Sobol64Generator):
        return GENERATOR_SOBOL64
    elif isinstance(generator, ScrambledSobol32Generator):
        return GENERATOR_SCRAMBLED_SOBOL32
    elif isinstance(generator, ScrambledSobol64Generator):
        return GENERATOR_SCRAMBLED_SOBOL64
    else:
        raise TypeError(f"Unsupported generator type: {type(generator)}")


def _generate_raw(
    generator,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    # Each generator has its own optimal block_size / num_warps; let it pick.
    raw = torch.empty(shape, dtype=torch.int32, device=device)
    generator.generate(raw)
    return raw


def _generate_raw64(
    generator,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    raw = torch.empty(shape, dtype=torch.int64, device=device)
    generator.generate_long_long(raw)
    return raw
