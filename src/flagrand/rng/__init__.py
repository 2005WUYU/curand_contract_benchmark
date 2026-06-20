from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Generator32(Protocol):
    @property
    def seed(self) -> int | None:
        ...

    @property
    def offset(self) -> int:
        ...

    @property
    def dimensions(self) -> int | None:
        ...

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class Generator64(Protocol):
    @property
    def seed(self) -> int | None:
        ...

    @property
    def offset(self) -> int:
        ...

    @property
    def dimensions(self) -> int | None:
        ...

    def generate_long_long(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ...


from flagrand.rng.mrg32k3a import Mrg32k3aGenerator  # noqa: E402
from flagrand.rng.mt19937 import Mt19937Generator  # noqa: E402
from flagrand.rng.mtgp32 import Mtgp32Generator  # noqa: E402
from flagrand.rng.philox import PhiloxGenerator  # noqa: E402
from flagrand.rng.scrambled_sobol32 import ScrambledSobol32Generator  # noqa: E402
from flagrand.rng.scrambled_sobol64 import ScrambledSobol64Generator  # noqa: E402
from flagrand.rng.sobol32 import Sobol32Generator  # noqa: E402
from flagrand.rng.sobol64 import Sobol64Generator  # noqa: E402
from flagrand.rng.xorwow import XorwowGenerator  # noqa: E402

__all__ = [
    "Generator32",
    "Generator64",
    "Mrg32k3aGenerator",
    "Mt19937Generator",
    "Mtgp32Generator",
    "PhiloxGenerator",
    "ScrambledSobol32Generator",
    "ScrambledSobol64Generator",
    "Sobol32Generator",
    "Sobol64Generator",
    "XorwowGenerator",
]
