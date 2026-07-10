from __future__ import annotations

from dataclasses import dataclass

import torch

from flagrand.rng._mtgp32_data import MTGP32_BLOCK_SIZE
from flagrand.rng._mtgp32_sequence import generate_mtgp32_contiguous


@dataclass
class Mtgp32Generator:
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
        offset_val = self.offset if offset is None else int(offset)
        if offset is not None and offset_val != self.offset:
            raise ValueError("MTGP32 explicit offset override is not supported.")
        seed_val = self.seed if seed is None else int(seed)
        block_size = int(kwargs.get("block_size", MTGP32_BLOCK_SIZE))
        if block_size != MTGP32_BLOCK_SIZE:
            raise ValueError("MTGP32 uses a fixed block_size=256 to preserve per-state dependency ordering.")
        num_warps = int(kwargs.get("num_warps", 8))

        n = out.numel()
        if n == 0:
            return out
        if seed is not None:
            raise ValueError("MTGP32 explicit seed override is not supported.")

        generate_mtgp32_contiguous(self, out, seed_val, offset_val, num_warps)
        self.offset = offset_val + n
        return out

    def generate_uniform(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if out.dtype != torch.float32:
            raise TypeError("MTGP32 generate_uniform requires float32 output.")
        offset_val = self.offset if offset is None else int(offset)
        if offset is not None and offset_val != self.offset:
            raise ValueError("MTGP32 explicit offset override is not supported.")
        seed_val = self.seed if seed is None else int(seed)
        block_size = int(kwargs.get("block_size", MTGP32_BLOCK_SIZE))
        if block_size != MTGP32_BLOCK_SIZE:
            raise ValueError("MTGP32 uses a fixed block_size=256 to preserve per-state dependency ordering.")
        num_warps = int(kwargs.get("num_warps", 8))

        n = out.numel()
        if n == 0:
            return out
        if seed is not None:
            raise ValueError("MTGP32 explicit seed override is not supported.")

        generate_mtgp32_contiguous(self, out, seed_val, offset_val, num_warps, output_mode=1)
        self.offset = offset_val + n
        return out
