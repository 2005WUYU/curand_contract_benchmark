from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flagrand.rng._sobol32_table import (
    launch_sobol32_normal_table,
    launch_sobol32_table,
    launch_sobol32_uniform_table,
)
from flagrand.rng._sobol_generator_utils import (
    normal_params,
    offset_from_kwargs,
    prepare_quasi_output,
)

_SOBOL32_SCRAMBLED_DV = torch.load(
    str(Path(__file__).parent / "data" / "scrambled_dv32.pt"), map_location="cpu"
)
_SOBOL32_SCRAMBLE_CONSTANTS = torch.load(
    str(Path(__file__).parent / "data" / "scramble_const32.pt"), map_location="cpu"
)
_SOBOL32_MAX_DIMENSIONS = 20000


@dataclass(frozen=True, slots=True)
class ScrambledSobol32Generator:
    dimensions: int = 1
    offset: int = 0

    @property
    def seed(self) -> None:
        return None

    def generate(
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
            dtype=torch.int32,
            dimensions=dimensions,
            max_dimensions=_SOBOL32_MAX_DIMENSIONS,
            offset=offset_val,
            op_name="ScrambledSobol32.generate",
        )
        if out.numel() == 0:
            return out
        launch_sobol32_table(
            out,
            _SOBOL32_SCRAMBLED_DV[:dimensions],
            dimensions=dimensions,
            offset=offset_val,
            block_size=block_size,
            num_warps=num_warps,
            scramble_constants=_SOBOL32_SCRAMBLE_CONSTANTS[:dimensions],
        )
        return out

    def generate_uniform(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        block_size = int(kwargs.get("block_size", 1024))
        num_warps = int(kwargs.get("num_warps", 8))
        offset_val = offset_from_kwargs(self.offset, kwargs)
        dimensions = int(self.dimensions)
        prepare_quasi_output(
            out,
            dtype=torch.float32,
            dimensions=dimensions,
            max_dimensions=_SOBOL32_MAX_DIMENSIONS,
            offset=offset_val,
            op_name="ScrambledSobol32.generate_uniform",
        )
        if out.numel() == 0:
            return out
        launch_sobol32_uniform_table(
            out,
            _SOBOL32_SCRAMBLED_DV[:dimensions],
            dimensions=dimensions,
            offset=offset_val,
            block_size=block_size,
            num_warps=num_warps,
            scramble_constants=_SOBOL32_SCRAMBLE_CONSTANTS[:dimensions],
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
            "ScrambledSobol32.generate_lognormal"
            if lognormal
            else "ScrambledSobol32.generate_normal"
        )
        prepare_quasi_output(
            out,
            dtype=torch.float32,
            dimensions=dimensions,
            max_dimensions=_SOBOL32_MAX_DIMENSIONS,
            offset=offset_val,
            op_name=op_name,
        )
        if out.numel() == 0:
            return out
        launch_sobol32_normal_table(
            out,
            _SOBOL32_SCRAMBLED_DV[:dimensions],
            dimensions=dimensions,
            offset=offset_val,
            mean=mean,
            stddev=stddev,
            lognormal=lognormal,
            block_size=block_size,
            num_warps=num_warps,
            scramble_constants=_SOBOL32_SCRAMBLE_CONSTANTS[:dimensions],
        )
        return out
