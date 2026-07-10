from __future__ import annotations

from dataclasses import dataclass

import torch

from flagrand.rng._mt19937_sequence import generate_mt19937_contiguous


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
        if out.numel() == 0:
            return out
        seed_val, offset_val = self._launch_args(out, seed=seed, offset=offset)
        num_warps = int(kwargs.get("num_warps", 1))
        generate_mt19937_contiguous(self, out, seed_val, offset_val, num_warps)
        self.offset = offset_val + out.numel()
        return out

    def generate_uniform(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if out.numel() == 0:
            return out
        if out.dtype != torch.float32:
            raise TypeError("MT19937 generate_uniform requires float32 output.")
        seed_val, offset_val = self._launch_args(out, seed=seed, offset=offset)
        num_warps = int(kwargs.get("num_warps", 1))
        generate_mt19937_contiguous(self, out, seed_val, offset_val, num_warps, output_mode=1)
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
