from __future__ import annotations

from typing import Callable

from contract_benchmark.capabilities import capability_matrix, optional_device_extension_status
from contract_benchmark.curand_adapter import curand_generate_by_distribution, make_curand_generator
from contract_benchmark.flagrand_adapter import (
    flagrand_generate_by_distribution,
    flagrand_generate_lognormal,
    flagrand_generate_normal,
    flagrand_generate_poisson,
    flagrand_generate_raw,
    flagrand_generate_uniform,
    make_flagrand_generator,
)
from contract_benchmark.generator_registry import GENERATOR_INFOS, GeneratorInfo

Operation = Callable[[], object]

__all__ = [
    "GENERATOR_INFOS",
    "GeneratorInfo",
    "Operation",
    "capability_matrix",
    "curand_generate_by_distribution",
    "flagrand_generate_by_distribution",
    "flagrand_generate_lognormal",
    "flagrand_generate_normal",
    "flagrand_generate_poisson",
    "flagrand_generate_raw",
    "flagrand_generate_uniform",
    "make_curand_generator",
    "make_flagrand_generator",
    "optional_device_extension_status",
]
