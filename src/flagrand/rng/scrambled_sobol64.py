from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import triton
import triton.language as tl

from flagrand._device import assert_tensor_device_supported, require_accelerator

_SOBOL64_SCRAMBLED_DV = torch.load(
    str(Path(__file__).parent / "data" / "scrambled_dv64.pt"), map_location="cpu"
)
_SOBOL64_SCRAMBLE_CONSTANTS = torch.load(
    str(Path(__file__).parent / "data" / "scramble_const64.pt"), map_location="cpu"
)

_SOBOL64_MAX_DIMENSIONS = 20000

# Triton GPU kernel (64-bit split into lo/hi, 32 bits each)

@triton.jit
def _scrambled_sobol64_generate(
    v0_lo, v1_lo, v2_lo, v3_lo, v4_lo, v5_lo, v6_lo, v7_lo,
    v8_lo, v9_lo, v10_lo, v11_lo, v12_lo, v13_lo, v14_lo, v15_lo,
    v16_lo, v17_lo, v18_lo, v19_lo, v20_lo, v21_lo, v22_lo, v23_lo,
    v24_lo, v25_lo, v26_lo, v27_lo, v28_lo, v29_lo, v30_lo, v31_lo,
    v0_hi, v1_hi, v2_hi, v3_hi, v4_hi, v5_hi, v6_hi, v7_hi,
    v8_hi, v9_hi, v10_hi, v11_hi, v12_hi, v13_hi, v14_hi, v15_hi,
    v16_hi, v17_hi, v18_hi, v19_hi, v20_hi, v21_hi, v22_hi, v23_hi,
    v24_hi, v25_hi, v26_hi, v27_hi, v28_hi, v29_hi, v30_hi, v31_hi,
    scramble_lo, scramble_hi,
    point_index,
    offset,
    BLOCK: tl.constexpr,
):
    # Gray code: g(i) = i XOR (i >> 1)
    idx = (point_index + offset).to(tl.uint64)
    gray_index = idx ^ (idx >> 1)

    result_lo = tl.zeros([BLOCK], dtype=tl.uint32)
    result_hi = tl.zeros([BLOCK], dtype=tl.uint32)

    # Low 32 bits
    b0 = ((gray_index >> 0) & 1) == 1
    result_lo = tl.where(b0, result_lo ^ tl.full([BLOCK], v0_lo, dtype=tl.uint32), result_lo)
    b1 = ((gray_index >> 1) & 1) == 1
    result_lo = tl.where(b1, result_lo ^ tl.full([BLOCK], v1_lo, dtype=tl.uint32), result_lo)
    b2 = ((gray_index >> 2) & 1) == 1
    result_lo = tl.where(b2, result_lo ^ tl.full([BLOCK], v2_lo, dtype=tl.uint32), result_lo)
    b3 = ((gray_index >> 3) & 1) == 1
    result_lo = tl.where(b3, result_lo ^ tl.full([BLOCK], v3_lo, dtype=tl.uint32), result_lo)
    b4 = ((gray_index >> 4) & 1) == 1
    result_lo = tl.where(b4, result_lo ^ tl.full([BLOCK], v4_lo, dtype=tl.uint32), result_lo)
    b5 = ((gray_index >> 5) & 1) == 1
    result_lo = tl.where(b5, result_lo ^ tl.full([BLOCK], v5_lo, dtype=tl.uint32), result_lo)
    b6 = ((gray_index >> 6) & 1) == 1
    result_lo = tl.where(b6, result_lo ^ tl.full([BLOCK], v6_lo, dtype=tl.uint32), result_lo)
    b7 = ((gray_index >> 7) & 1) == 1
    result_lo = tl.where(b7, result_lo ^ tl.full([BLOCK], v7_lo, dtype=tl.uint32), result_lo)
    b8 = ((gray_index >> 8) & 1) == 1
    result_lo = tl.where(b8, result_lo ^ tl.full([BLOCK], v8_lo, dtype=tl.uint32), result_lo)
    b9 = ((gray_index >> 9) & 1) == 1
    result_lo = tl.where(b9, result_lo ^ tl.full([BLOCK], v9_lo, dtype=tl.uint32), result_lo)
    b10 = ((gray_index >> 10) & 1) == 1
    result_lo = tl.where(b10, result_lo ^ tl.full([BLOCK], v10_lo, dtype=tl.uint32), result_lo)
    b11 = ((gray_index >> 11) & 1) == 1
    result_lo = tl.where(b11, result_lo ^ tl.full([BLOCK], v11_lo, dtype=tl.uint32), result_lo)
    b12 = ((gray_index >> 12) & 1) == 1
    result_lo = tl.where(b12, result_lo ^ tl.full([BLOCK], v12_lo, dtype=tl.uint32), result_lo)
    b13 = ((gray_index >> 13) & 1) == 1
    result_lo = tl.where(b13, result_lo ^ tl.full([BLOCK], v13_lo, dtype=tl.uint32), result_lo)
    b14 = ((gray_index >> 14) & 1) == 1
    result_lo = tl.where(b14, result_lo ^ tl.full([BLOCK], v14_lo, dtype=tl.uint32), result_lo)
    b15 = ((gray_index >> 15) & 1) == 1
    result_lo = tl.where(b15, result_lo ^ tl.full([BLOCK], v15_lo, dtype=tl.uint32), result_lo)
    b16 = ((gray_index >> 16) & 1) == 1
    result_lo = tl.where(b16, result_lo ^ tl.full([BLOCK], v16_lo, dtype=tl.uint32), result_lo)
    b17 = ((gray_index >> 17) & 1) == 1
    result_lo = tl.where(b17, result_lo ^ tl.full([BLOCK], v17_lo, dtype=tl.uint32), result_lo)
    b18 = ((gray_index >> 18) & 1) == 1
    result_lo = tl.where(b18, result_lo ^ tl.full([BLOCK], v18_lo, dtype=tl.uint32), result_lo)
    b19 = ((gray_index >> 19) & 1) == 1
    result_lo = tl.where(b19, result_lo ^ tl.full([BLOCK], v19_lo, dtype=tl.uint32), result_lo)
    b20 = ((gray_index >> 20) & 1) == 1
    result_lo = tl.where(b20, result_lo ^ tl.full([BLOCK], v20_lo, dtype=tl.uint32), result_lo)
    b21 = ((gray_index >> 21) & 1) == 1
    result_lo = tl.where(b21, result_lo ^ tl.full([BLOCK], v21_lo, dtype=tl.uint32), result_lo)
    b22 = ((gray_index >> 22) & 1) == 1
    result_lo = tl.where(b22, result_lo ^ tl.full([BLOCK], v22_lo, dtype=tl.uint32), result_lo)
    b23 = ((gray_index >> 23) & 1) == 1
    result_lo = tl.where(b23, result_lo ^ tl.full([BLOCK], v23_lo, dtype=tl.uint32), result_lo)
    b24 = ((gray_index >> 24) & 1) == 1
    result_lo = tl.where(b24, result_lo ^ tl.full([BLOCK], v24_lo, dtype=tl.uint32), result_lo)
    b25 = ((gray_index >> 25) & 1) == 1
    result_lo = tl.where(b25, result_lo ^ tl.full([BLOCK], v25_lo, dtype=tl.uint32), result_lo)
    b26 = ((gray_index >> 26) & 1) == 1
    result_lo = tl.where(b26, result_lo ^ tl.full([BLOCK], v26_lo, dtype=tl.uint32), result_lo)
    b27 = ((gray_index >> 27) & 1) == 1
    result_lo = tl.where(b27, result_lo ^ tl.full([BLOCK], v27_lo, dtype=tl.uint32), result_lo)
    b28 = ((gray_index >> 28) & 1) == 1
    result_lo = tl.where(b28, result_lo ^ tl.full([BLOCK], v28_lo, dtype=tl.uint32), result_lo)
    b29 = ((gray_index >> 29) & 1) == 1
    result_lo = tl.where(b29, result_lo ^ tl.full([BLOCK], v29_lo, dtype=tl.uint32), result_lo)
    b30 = ((gray_index >> 30) & 1) == 1
    result_lo = tl.where(b30, result_lo ^ tl.full([BLOCK], v30_lo, dtype=tl.uint32), result_lo)
    b31 = ((gray_index >> 31) & 1) == 1
    result_lo = tl.where(b31, result_lo ^ tl.full([BLOCK], v31_lo, dtype=tl.uint32), result_lo)

    # High 32 bits
    bh0 = ((gray_index >> 0) & 1) == 1
    result_hi = tl.where(bh0, result_hi ^ tl.full([BLOCK], v0_hi, dtype=tl.uint32), result_hi)
    bh1 = ((gray_index >> 1) & 1) == 1
    result_hi = tl.where(bh1, result_hi ^ tl.full([BLOCK], v1_hi, dtype=tl.uint32), result_hi)
    bh2 = ((gray_index >> 2) & 1) == 1
    result_hi = tl.where(bh2, result_hi ^ tl.full([BLOCK], v2_hi, dtype=tl.uint32), result_hi)
    bh3 = ((gray_index >> 3) & 1) == 1
    result_hi = tl.where(bh3, result_hi ^ tl.full([BLOCK], v3_hi, dtype=tl.uint32), result_hi)
    bh4 = ((gray_index >> 4) & 1) == 1
    result_hi = tl.where(bh4, result_hi ^ tl.full([BLOCK], v4_hi, dtype=tl.uint32), result_hi)
    bh5 = ((gray_index >> 5) & 1) == 1
    result_hi = tl.where(bh5, result_hi ^ tl.full([BLOCK], v5_hi, dtype=tl.uint32), result_hi)
    bh6 = ((gray_index >> 6) & 1) == 1
    result_hi = tl.where(bh6, result_hi ^ tl.full([BLOCK], v6_hi, dtype=tl.uint32), result_hi)
    bh7 = ((gray_index >> 7) & 1) == 1
    result_hi = tl.where(bh7, result_hi ^ tl.full([BLOCK], v7_hi, dtype=tl.uint32), result_hi)
    bh8 = ((gray_index >> 8) & 1) == 1
    result_hi = tl.where(bh8, result_hi ^ tl.full([BLOCK], v8_hi, dtype=tl.uint32), result_hi)
    bh9 = ((gray_index >> 9) & 1) == 1
    result_hi = tl.where(bh9, result_hi ^ tl.full([BLOCK], v9_hi, dtype=tl.uint32), result_hi)
    bh10 = ((gray_index >> 10) & 1) == 1
    result_hi = tl.where(bh10, result_hi ^ tl.full([BLOCK], v10_hi, dtype=tl.uint32), result_hi)
    bh11 = ((gray_index >> 11) & 1) == 1
    result_hi = tl.where(bh11, result_hi ^ tl.full([BLOCK], v11_hi, dtype=tl.uint32), result_hi)
    bh12 = ((gray_index >> 12) & 1) == 1
    result_hi = tl.where(bh12, result_hi ^ tl.full([BLOCK], v12_hi, dtype=tl.uint32), result_hi)
    bh13 = ((gray_index >> 13) & 1) == 1
    result_hi = tl.where(bh13, result_hi ^ tl.full([BLOCK], v13_hi, dtype=tl.uint32), result_hi)
    bh14 = ((gray_index >> 14) & 1) == 1
    result_hi = tl.where(bh14, result_hi ^ tl.full([BLOCK], v14_hi, dtype=tl.uint32), result_hi)
    bh15 = ((gray_index >> 15) & 1) == 1
    result_hi = tl.where(bh15, result_hi ^ tl.full([BLOCK], v15_hi, dtype=tl.uint32), result_hi)
    bh16 = ((gray_index >> 16) & 1) == 1
    result_hi = tl.where(bh16, result_hi ^ tl.full([BLOCK], v16_hi, dtype=tl.uint32), result_hi)
    bh17 = ((gray_index >> 17) & 1) == 1
    result_hi = tl.where(bh17, result_hi ^ tl.full([BLOCK], v17_hi, dtype=tl.uint32), result_hi)
    bh18 = ((gray_index >> 18) & 1) == 1
    result_hi = tl.where(bh18, result_hi ^ tl.full([BLOCK], v18_hi, dtype=tl.uint32), result_hi)
    bh19 = ((gray_index >> 19) & 1) == 1
    result_hi = tl.where(bh19, result_hi ^ tl.full([BLOCK], v19_hi, dtype=tl.uint32), result_hi)
    bh20 = ((gray_index >> 20) & 1) == 1
    result_hi = tl.where(bh20, result_hi ^ tl.full([BLOCK], v20_hi, dtype=tl.uint32), result_hi)
    bh21 = ((gray_index >> 21) & 1) == 1
    result_hi = tl.where(bh21, result_hi ^ tl.full([BLOCK], v21_hi, dtype=tl.uint32), result_hi)
    bh22 = ((gray_index >> 22) & 1) == 1
    result_hi = tl.where(bh22, result_hi ^ tl.full([BLOCK], v22_hi, dtype=tl.uint32), result_hi)
    bh23 = ((gray_index >> 23) & 1) == 1
    result_hi = tl.where(bh23, result_hi ^ tl.full([BLOCK], v23_hi, dtype=tl.uint32), result_hi)
    bh24 = ((gray_index >> 24) & 1) == 1
    result_hi = tl.where(bh24, result_hi ^ tl.full([BLOCK], v24_hi, dtype=tl.uint32), result_hi)
    bh25 = ((gray_index >> 25) & 1) == 1
    result_hi = tl.where(bh25, result_hi ^ tl.full([BLOCK], v25_hi, dtype=tl.uint32), result_hi)
    bh26 = ((gray_index >> 26) & 1) == 1
    result_hi = tl.where(bh26, result_hi ^ tl.full([BLOCK], v26_hi, dtype=tl.uint32), result_hi)
    bh27 = ((gray_index >> 27) & 1) == 1
    result_hi = tl.where(bh27, result_hi ^ tl.full([BLOCK], v27_hi, dtype=tl.uint32), result_hi)
    bh28 = ((gray_index >> 28) & 1) == 1
    result_hi = tl.where(bh28, result_hi ^ tl.full([BLOCK], v28_hi, dtype=tl.uint32), result_hi)
    bh29 = ((gray_index >> 29) & 1) == 1
    result_hi = tl.where(bh29, result_hi ^ tl.full([BLOCK], v29_hi, dtype=tl.uint32), result_hi)
    bh30 = ((gray_index >> 30) & 1) == 1
    result_hi = tl.where(bh30, result_hi ^ tl.full([BLOCK], v30_hi, dtype=tl.uint32), result_hi)
    bh31 = ((gray_index >> 31) & 1) == 1
    result_hi = tl.where(bh31, result_hi ^ tl.full([BLOCK], v31_hi, dtype=tl.uint32), result_hi)

    # XOR scramble
    result_lo = result_lo ^ tl.full([BLOCK], scramble_lo, dtype=tl.uint32)
    result_hi = result_hi ^ tl.full([BLOCK], scramble_hi, dtype=tl.uint32)

    return result_lo, result_hi

@triton.jit
def _scrambled_sobol64_single_dim_kernel(
    out_int32_ptr,
    v0_lo, v1_lo, v2_lo, v3_lo, v4_lo, v5_lo, v6_lo, v7_lo,
    v8_lo, v9_lo, v10_lo, v11_lo, v12_lo, v13_lo, v14_lo, v15_lo,
    v16_lo, v17_lo, v18_lo, v19_lo, v20_lo, v21_lo, v22_lo, v23_lo,
    v24_lo, v25_lo, v26_lo, v27_lo, v28_lo, v29_lo, v30_lo, v31_lo,
    v0_hi, v1_hi, v2_hi, v3_hi, v4_hi, v5_hi, v6_hi, v7_hi,
    v8_hi, v9_hi, v10_hi, v11_hi, v12_hi, v13_hi, v14_hi, v15_hi,
    v16_hi, v17_hi, v18_hi, v19_hi, v20_hi, v21_hi, v22_hi, v23_hi,
    v24_hi, v25_hi, v26_hi, v27_hi, v28_hi, v29_hi, v30_hi, v31_hi,
    scramble_lo, scramble_hi,
    n_points,
    offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_points

    result_lo, result_hi = _scrambled_sobol64_generate(
        v0_lo, v1_lo, v2_lo, v3_lo, v4_lo, v5_lo, v6_lo, v7_lo,
        v8_lo, v9_lo, v10_lo, v11_lo, v12_lo, v13_lo, v14_lo, v15_lo,
        v16_lo, v17_lo, v18_lo, v19_lo, v20_lo, v21_lo, v22_lo, v23_lo,
        v24_lo, v25_lo, v26_lo, v27_lo, v28_lo, v29_lo, v30_lo, v31_lo,
        v0_hi, v1_hi, v2_hi, v3_hi, v4_hi, v5_hi, v6_hi, v7_hi,
        v8_hi, v9_hi, v10_hi, v11_hi, v12_hi, v13_hi, v14_hi, v15_hi,
        v16_hi, v17_hi, v18_hi, v19_hi, v20_hi, v21_hi, v22_hi, v23_hi,
        v24_hi, v25_hi, v26_hi, v27_hi, v28_hi, v29_hi, v30_hi, v31_hi,
        scramble_lo, scramble_hi,
        offs, offset, BLOCK=BLOCK,
    )

    base = 2 * offs
    tl.store(out_int32_ptr + base + 0, result_lo.to(tl.int32, bitcast=True), mask=mask)
    tl.store(out_int32_ptr + base + 1, result_hi.to(tl.int32, bitcast=True), mask=mask)

# Host-side launch logic

def _launch_scrambled_sobol64_dim(
    out_int32_dim: torch.Tensor,
    dv_lo: list[int],
    dv_hi: list[int],
    scramble_lo: int,
    scramble_hi: int,
    n_points: int,
    offset: int,
    block_size: int,
    num_warps: int,
) -> None:
    grid = (triton.cdiv(n_points, block_size),)
    _scrambled_sobol64_single_dim_kernel[grid](
        out_int32_dim,
        dv_lo[0], dv_lo[1], dv_lo[2], dv_lo[3],
        dv_lo[4], dv_lo[5], dv_lo[6], dv_lo[7],
        dv_lo[8], dv_lo[9], dv_lo[10], dv_lo[11],
        dv_lo[12], dv_lo[13], dv_lo[14], dv_lo[15],
        dv_lo[16], dv_lo[17], dv_lo[18], dv_lo[19],
        dv_lo[20], dv_lo[21], dv_lo[22], dv_lo[23],
        dv_lo[24], dv_lo[25], dv_lo[26], dv_lo[27],
        dv_lo[28], dv_lo[29], dv_lo[30], dv_lo[31],
        dv_hi[0], dv_hi[1], dv_hi[2], dv_hi[3],
        dv_hi[4], dv_hi[5], dv_hi[6], dv_hi[7],
        dv_hi[8], dv_hi[9], dv_hi[10], dv_hi[11],
        dv_hi[12], dv_hi[13], dv_hi[14], dv_hi[15],
        dv_hi[16], dv_hi[17], dv_hi[18], dv_hi[19],
        dv_hi[20], dv_hi[21], dv_hi[22], dv_hi[23],
        dv_hi[24], dv_hi[25], dv_hi[26], dv_hi[27],
        dv_hi[28], dv_hi[29], dv_hi[30], dv_hi[31],
        scramble_lo,
        scramble_hi,
        n_points,
        offset,
        BLOCK=block_size,
        num_warps=num_warps,
    )

# Public API

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
        block_size = kwargs.get("block_size", 256)
        num_warps = kwargs.get("num_warps", 8)

        gen = self
        if offset is not None:
            gen = replace(gen, offset=offset)
        offset_val = int(gen.offset)

        require_accelerator()
        if out.dtype != torch.int64:
            raise TypeError("ScrambledSobol64: only uint64 (int64) output is supported.")
        assert_tensor_device_supported(out, op_name="ScrambledSobol64.generate_long_long")

        n = out.numel()
        if n == 0:
            return out

        dimensions = int(gen.dimensions)
        if dimensions < 1 or dimensions > _SOBOL64_MAX_DIMENSIONS:
            raise ValueError(
                f"ScrambledSobol64: dimensions must be between 1 and {_SOBOL64_MAX_DIMENSIONS}, got {dimensions}."
            )
        if n % dimensions != 0:
            raise ValueError(
                f"ScrambledSobol64: element count ({n}) must be a multiple of dimensions ({dimensions})."
            )
        if offset_val < 0:
            raise ValueError(f"ScrambledSobol64: offset must be >= 0, got {offset_val}.")

        n_points_per_dim = n // dimensions
        sdvs = _SOBOL64_SCRAMBLED_DV[:dimensions]
        scrambles = _SOBOL64_SCRAMBLE_CONSTANTS[:dimensions]

        # write directly into out, viewed as int32 pairs
        out_int32 = out.view(-1).view(torch.int32)

        for d in range(dimensions):
            base_int32 = d * (2 * n_points_per_dim)
            out_int32_dim = out_int32[base_int32 : base_int32 + 2 * n_points_per_dim]
            sdv64 = sdvs[d].tolist()
            sdv_lo = [v & 0xFFFFFFFF for v in sdv64[:32]]
            sdv_hi = [v >> 32 for v in sdv64[:32]]
            sc64 = int(scrambles[d].item())
            _launch_scrambled_sobol64_dim(
                out_int32_dim, sdv_lo, sdv_hi,
                sc64 & 0xFFFFFFFF, sc64 >> 32,
                n_points_per_dim, offset_val, block_size, num_warps,
            )

        return out

