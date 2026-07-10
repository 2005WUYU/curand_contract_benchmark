from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flagrand.rng._sobol64_table import (
    launch_sobol64_normal_table,
    launch_sobol64_table,
    launch_sobol64_uniform_table,
)
from flagrand.rng._sobol_generator_utils import (
    normal_params,
    offset_from_kwargs,
    prepare_quasi_output,
)

_SOBOL64_SCRAMBLED_DV = torch.load(
    str(Path(__file__).parent / "data" / "scrambled_dv64.pt"), map_location="cpu"
)
_SOBOL64_SCRAMBLE_CONSTANTS = torch.load(
    str(Path(__file__).parent / "data" / "scramble_const64.pt"), map_location="cpu"
)
_SOBOL64_MAX_DIMENSIONS = 20000


@dataclass(frozen=True, slots=True)
class ScrambledSobol64Generator:
    dimensions: int = 1
    offset: int = 0

    @property
    def seed(self) -> None:
        return None

    def generate_long_long(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        block_size = int(kwargs.get("block_size", 1024))
        num_warps = int(kwargs.get("num_warps", 8))
        offset_val = self.offset if offset is None else int(offset)
        dimensions = int(self.dimensions)
        prepare_quasi_output(
            out,
            dtype=torch.int64,
            dimensions=dimensions,
            max_dimensions=_SOBOL64_MAX_DIMENSIONS,
            offset=offset_val,
            op_name="ScrambledSobol64.generate_long_long",
        )
        if out.numel() == 0:
            return out
        launch_sobol64_table(
            out,
            _SOBOL64_SCRAMBLED_DV[:dimensions],
            dimensions=dimensions,
            offset=offset_val,
            block_size=block_size,
            num_warps=num_warps,
            scramble_constants=_SOBOL64_SCRAMBLE_CONSTANTS[:dimensions],
        )
        return out

    def generate_uniform(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        block_size = int(kwargs.get("block_size", 1024))
        num_warps = int(kwargs.get("num_warps", 8))
        offset_val = offset_from_kwargs(self.offset, kwargs)
        dimensions = int(self.dimensions)
        prepare_quasi_output(
            out,
            dtype=torch.float64,
            dimensions=dimensions,
            max_dimensions=_SOBOL64_MAX_DIMENSIONS,
            offset=offset_val,
            op_name="ScrambledSobol64.generate_uniform",
        )
        if out.numel() == 0:
            return out
        launch_sobol64_uniform_table(
            out,
            _SOBOL64_SCRAMBLED_DV[:dimensions],
            dimensions=dimensions,
            offset=offset_val,
            block_size=block_size,
            num_warps=num_warps,
            scramble_constants=_SOBOL64_SCRAMBLE_CONSTANTS[:dimensions],
        )
        return out

    def generate_normal(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self._generate_normal_like(out, lognormal=False, **kwargs)

    def generate_lognormal(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self._generate_normal_like(out, lognormal=True, **kwargs)

    def _generate_normal_like(
        self,
        out: torch.Tensor,
        *,
        lognormal: bool,
        **kwargs: object,
    ) -> torch.Tensor:
        block_size = int(kwargs.get("block_size", 1024))
        num_warps = int(kwargs.get("num_warps", 8))
        offset_val = offset_from_kwargs(self.offset, kwargs)
        mean, stddev = normal_params(kwargs)
        dimensions = int(self.dimensions)
        op_name = (
            "ScrambledSobol64.generate_lognormal"
            if lognormal
            else "ScrambledSobol64.generate_normal"
        )
        prepare_quasi_output(
            out,
            dtype=torch.float64,
            dimensions=dimensions,
            max_dimensions=_SOBOL64_MAX_DIMENSIONS,
            offset=offset_val,
            op_name=op_name,
        )
        if out.numel() == 0:
            return out
        launch_sobol64_normal_table(
            out,
            _SOBOL64_SCRAMBLED_DV[:dimensions],
            dimensions=dimensions,
            offset=offset_val,
            mean=mean,
            stddev=stddev,
            lognormal=lognormal,
            block_size=block_size,
            num_warps=num_warps,
            scramble_constants=_SOBOL64_SCRAMBLE_CONSTANTS[:dimensions],
        )
        return out
