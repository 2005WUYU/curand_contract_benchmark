from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import triton
import triton.language as tl

from flagrand._device import assert_tensor_device_supported, require_accelerator

_SOBOL32_DIRECTION_VECTORS = torch.load(
    str(Path(__file__).parent / "data" / "sobol_dv32.pt"), map_location="cpu"
)

_SOBOL32_MAX_DIMENSIONS = 20000


@triton.jit
def _sobol32_generate(
    v0, v1, v2, v3, v4, v5, v6, v7,
    v8, v9, v10, v11, v12, v13, v14, v15,
    v16, v17, v18, v19, v20, v21, v22, v23,
    v24, v25, v26, v27, v28, v29, v30, v31,
    point_index,
    offset,
):
    # Gray code: g(i) = i XOR (i >> 1)
    idx = point_index + offset
    gray_index = idx ^ (idx >> 1)

    result = tl.zeros_like(idx)

    b0 = ((gray_index >> 0) & 1) == 1
    result = tl.where(b0, result ^ v0 + tl.zeros_like(idx), result)
    b1 = ((gray_index >> 1) & 1) == 1
    result = tl.where(b1, result ^ v1 + tl.zeros_like(idx), result)
    b2 = ((gray_index >> 2) & 1) == 1
    result = tl.where(b2, result ^ v2 + tl.zeros_like(idx), result)
    b3 = ((gray_index >> 3) & 1) == 1
    result = tl.where(b3, result ^ v3 + tl.zeros_like(idx), result)
    b4 = ((gray_index >> 4) & 1) == 1
    result = tl.where(b4, result ^ v4 + tl.zeros_like(idx), result)
    b5 = ((gray_index >> 5) & 1) == 1
    result = tl.where(b5, result ^ v5 + tl.zeros_like(idx), result)
    b6 = ((gray_index >> 6) & 1) == 1
    result = tl.where(b6, result ^ v6 + tl.zeros_like(idx), result)
    b7 = ((gray_index >> 7) & 1) == 1
    result = tl.where(b7, result ^ v7 + tl.zeros_like(idx), result)
    b8 = ((gray_index >> 8) & 1) == 1
    result = tl.where(b8, result ^ v8 + tl.zeros_like(idx), result)
    b9 = ((gray_index >> 9) & 1) == 1
    result = tl.where(b9, result ^ v9 + tl.zeros_like(idx), result)
    b10 = ((gray_index >> 10) & 1) == 1
    result = tl.where(b10, result ^ v10 + tl.zeros_like(idx), result)
    b11 = ((gray_index >> 11) & 1) == 1
    result = tl.where(b11, result ^ v11 + tl.zeros_like(idx), result)
    b12 = ((gray_index >> 12) & 1) == 1
    result = tl.where(b12, result ^ v12 + tl.zeros_like(idx), result)
    b13 = ((gray_index >> 13) & 1) == 1
    result = tl.where(b13, result ^ v13 + tl.zeros_like(idx), result)
    b14 = ((gray_index >> 14) & 1) == 1
    result = tl.where(b14, result ^ v14 + tl.zeros_like(idx), result)
    b15 = ((gray_index >> 15) & 1) == 1
    result = tl.where(b15, result ^ v15 + tl.zeros_like(idx), result)
    b16 = ((gray_index >> 16) & 1) == 1
    result = tl.where(b16, result ^ v16 + tl.zeros_like(idx), result)
    b17 = ((gray_index >> 17) & 1) == 1
    result = tl.where(b17, result ^ v17 + tl.zeros_like(idx), result)
    b18 = ((gray_index >> 18) & 1) == 1
    result = tl.where(b18, result ^ v18 + tl.zeros_like(idx), result)
    b19 = ((gray_index >> 19) & 1) == 1
    result = tl.where(b19, result ^ v19 + tl.zeros_like(idx), result)
    b20 = ((gray_index >> 20) & 1) == 1
    result = tl.where(b20, result ^ v20 + tl.zeros_like(idx), result)
    b21 = ((gray_index >> 21) & 1) == 1
    result = tl.where(b21, result ^ v21 + tl.zeros_like(idx), result)
    b22 = ((gray_index >> 22) & 1) == 1
    result = tl.where(b22, result ^ v22 + tl.zeros_like(idx), result)
    b23 = ((gray_index >> 23) & 1) == 1
    result = tl.where(b23, result ^ v23 + tl.zeros_like(idx), result)
    b24 = ((gray_index >> 24) & 1) == 1
    result = tl.where(b24, result ^ v24 + tl.zeros_like(idx), result)
    b25 = ((gray_index >> 25) & 1) == 1
    result = tl.where(b25, result ^ v25 + tl.zeros_like(idx), result)
    b26 = ((gray_index >> 26) & 1) == 1
    result = tl.where(b26, result ^ v26 + tl.zeros_like(idx), result)
    b27 = ((gray_index >> 27) & 1) == 1
    result = tl.where(b27, result ^ v27 + tl.zeros_like(idx), result)
    b28 = ((gray_index >> 28) & 1) == 1
    result = tl.where(b28, result ^ v28 + tl.zeros_like(idx), result)
    b29 = ((gray_index >> 29) & 1) == 1
    result = tl.where(b29, result ^ v29 + tl.zeros_like(idx), result)
    b30 = ((gray_index >> 30) & 1) == 1
    result = tl.where(b30, result ^ v30 + tl.zeros_like(idx), result)
    b31 = ((gray_index >> 31) & 1) == 1
    result = tl.where(b31, result ^ v31 + tl.zeros_like(idx), result)

    return result


@triton.jit
def _sobol32_single_dim_kernel(
    out_ptr,
    v0, v1, v2, v3, v4, v5, v6, v7,
    v8, v9, v10, v11, v12, v13, v14, v15,
    v16, v17, v18, v19, v20, v21, v22, v23,
    v24, v25, v26, v27, v28, v29, v30, v31,
    n_points,
    offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_points

    result = _sobol32_generate(
        v0, v1, v2, v3, v4, v5, v6, v7,
        v8, v9, v10, v11, v12, v13, v14, v15,
        v16, v17, v18, v19, v20, v21, v22, v23,
        v24, v25, v26, v27, v28, v29, v30, v31,
        offs, offset,
    )

    result_u32 = result.to(tl.uint32)
    tl.store(out_ptr + offs, result_u32.to(tl.int32, bitcast=True), mask=mask)


def _launch_sobol32_dim(
    out_dim: torch.Tensor,
    dv: list[int],
    n_points: int,
    offset: int,
    block_size: int,
    num_warps: int,
) -> None:
    grid = (triton.cdiv(n_points, block_size),)
    _sobol32_single_dim_kernel[grid](
        out_dim,
        dv[0], dv[1], dv[2], dv[3],
        dv[4], dv[5], dv[6], dv[7],
        dv[8], dv[9], dv[10], dv[11],
        dv[12], dv[13], dv[14], dv[15],
        dv[16], dv[17], dv[18], dv[19],
        dv[20], dv[21], dv[22], dv[23],
        dv[24], dv[25], dv[26], dv[27],
        dv[28], dv[29], dv[30], dv[31],
        n_points,
        offset,
        BLOCK=block_size,
        num_warps=num_warps,
    )


@dataclass(frozen=True, slots=True)
class Sobol32Generator:
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
        block_size = kwargs.get("block_size", 256)
        num_warps = kwargs.get("num_warps", 8)

        gen = self
        if offset is not None:
            gen = replace(gen, offset=offset)
        offset_val = int(gen.offset)

        require_accelerator()
        if out.dtype != torch.int32:
            raise TypeError("Sobol32: only uint32 (int32) output is supported.")
        assert_tensor_device_supported(out, op_name="Sobol32.generate")

        n = out.numel()
        if n == 0:
            return out

        dimensions = int(gen.dimensions)
        if dimensions < 1 or dimensions > _SOBOL32_MAX_DIMENSIONS:
            raise ValueError(
                f"Sobol32: dimensions must be between 1 and {_SOBOL32_MAX_DIMENSIONS}, got {dimensions}."
            )
        if n % dimensions != 0:
            raise ValueError(
                f"Sobol32: element count ({n}) must be a multiple of dimensions ({dimensions})."
            )
        if offset_val < 0:
            raise ValueError(f"Sobol32: offset must be >= 0, got {offset_val}.")

        n_points_per_dim = n // dimensions
        dvs = _SOBOL32_DIRECTION_VECTORS[:dimensions]
        out_flat = out.view(-1)

        for d in range(dimensions):
            out_dim = out_flat[d * n_points_per_dim : (d + 1) * n_points_per_dim]
            _launch_sobol32_dim(
                out_dim,
                dvs[d].tolist(),
                n_points_per_dim,
                offset_val,
                block_size,
                num_warps,
            )

        return out
