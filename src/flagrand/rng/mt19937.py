from __future__ import annotations

from dataclasses import dataclass

import torch

from flagrand.rng._mt19937_sequence import generate_mt19937_contiguous
from flagrand.rng._stateful_output import (
    RAW_OUTPUT,
    UNIFORM_OUTPUT,
    StatefulOutput,
    normal_output,
    poisson_output,
    small_poisson_max_k,
)


@dataclass
class Mt19937Generator:
    seed: int = 0
    offset: int = 0

    @property
    def dimensions(self) -> None:
        return None

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if out.dtype != torch.int32:
            raise TypeError("MT19937 generate requires int32 output.")
        return self._generate_output(out, RAW_OUTPUT, seed=seed, offset=offset, kwargs=kwargs)

    def generate_uniform(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if out.dtype != torch.float32:
            raise TypeError("MT19937 generate_uniform requires float32 output.")
        return self._generate_output(out, UNIFORM_OUTPUT, seed=seed, offset=offset, kwargs=kwargs)

    def generate_normal(
        self,
        out: torch.Tensor,
        *,
        mean: float,
        stddev: float,
        **kwargs: object,
    ) -> torch.Tensor:
        _require_float32_pairs(out, "MT19937 generate_normal")
        return self._generate_output(
            out,
            normal_output(mean, stddev, lognormal=False),
            seed=None,
            offset=None,
            kwargs=kwargs,
        )

    def generate_lognormal(
        self,
        out: torch.Tensor,
        *,
        mean: float,
        stddev: float,
        **kwargs: object,
    ) -> torch.Tensor:
        _require_float32_pairs(out, "MT19937 generate_lognormal")
        return self._generate_output(
            out,
            normal_output(mean, stddev, lognormal=True),
            seed=None,
            offset=None,
            kwargs=kwargs,
        )

    def generate_poisson(
        self,
        out: torch.Tensor,
        *,
        lambda_val: float,
        **kwargs: object,
    ) -> torch.Tensor:
        if out.dtype != torch.int32:
            raise TypeError("MT19937 generate_poisson requires int32 output.")
        if lambda_val <= 0:
            raise ValueError(f"MT19937 generate_poisson requires lambda > 0, got {lambda_val}.")
        if lambda_val >= 30.0 and out.numel() % 2:
            raise ValueError("MT19937 large-lambda Poisson requires an even element count.")
        max_k = small_poisson_max_k(lambda_val) if lambda_val < 30.0 else 0
        return self._generate_output(
            out,
            poisson_output(lambda_val, max_k),
            seed=None,
            offset=None,
            kwargs=kwargs,
        )

    def _generate_output(
        self,
        out: torch.Tensor,
        output: StatefulOutput,
        *,
        seed: int | None,
        offset: int | None,
        kwargs: dict[str, object],
    ) -> torch.Tensor:
        if out.numel() == 0:
            return out
        self._launch_args(out, seed=seed, offset=offset)
        num_warps = int(kwargs.get("num_warps", 4))
        return self._generate_prepared(out, output, num_warps=num_warps)

    def _generate_prepared(
        self,
        out: torch.Tensor,
        output: StatefulOutput,
        *,
        num_warps: int,
    ) -> torch.Tensor:
        offset_val = self.offset
        generate_mt19937_contiguous(
            self,
            out,
            self.seed,
            offset_val,
            num_warps,
            output=output,
        )
        self.offset = offset_val + out.numel()
        return out

    def _launch_args(
        self,
        out: torch.Tensor,
        *,
        seed: int | None,
        offset: int | None,
    ) -> tuple[int, int]:
        n = out.numel()
        if n == 0:
            return self.seed, self.offset if offset is None else int(offset)
        if seed is not None:
            raise ValueError("MT19937 explicit seed override is not supported.")

        offset_val = self.offset if offset is None else int(offset)
        if offset is not None and offset_val != self.offset:
            raise ValueError("MT19937 explicit offset override is not supported.")
        if offset_val != 0 and not hasattr(self, "_working_state"):
            raise ValueError(f"MT19937 does not support non-zero initial offset, got {offset_val}.")
        return self.seed, offset_val


def _require_float32_pairs(out: torch.Tensor, op_name: str) -> None:
    if out.dtype != torch.float32:
        raise TypeError(f"{op_name} requires float32 output.")
    if out.numel() % 2:
        raise ValueError(f"{op_name} requires an even element count, got {out.numel()}.")
