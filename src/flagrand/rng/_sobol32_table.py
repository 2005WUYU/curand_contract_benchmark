from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.transforms import uniform_to_normal_icdf_f32
from flagrand.rng._sobol_chunk_tables import launch_plan32
from flagrand.runtime import CachedKernelLauncher


@triton.jit
def _sobol32_chunk_table_kernel(
    out_ptr,
    chunk_ptr,
    scramble_ptr,
    points_per_dim,
    offset,
    CHUNKS: tl.constexpr,
    HAS_SCRAMBLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dim = tl.program_id(1)
    point = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = point < points_per_dim
    gray = (point + offset) ^ ((point + offset) >> 1)
    result = tl.full((BLOCK,), 0, tl.uint32)

    for chunk in tl.static_range(CHUNKS):
        byte = ((gray >> (chunk * 8)) & 0xFF).to(tl.int32)
        part = tl.load(chunk_ptr + (dim * CHUNKS + chunk) * 256 + byte, mask=mask, other=0)
        result = result ^ part.to(tl.uint32, bitcast=True)

    if HAS_SCRAMBLE:
        result = result ^ tl.load(scramble_ptr + dim).to(tl.uint32, bitcast=True)

    tl.store(out_ptr + dim * points_per_dim + point, result.to(tl.int32, bitcast=True), mask=mask)


@triton.jit
def _sobol32_uniform_chunk_table_kernel(
    out_ptr,
    chunk_ptr,
    scramble_ptr,
    points_per_dim,
    offset,
    CHUNKS: tl.constexpr,
    HAS_SCRAMBLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dim = tl.program_id(1)
    point = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = point < points_per_dim
    gray = (point + offset) ^ ((point + offset) >> 1)
    result = tl.full((BLOCK,), 0, tl.uint32)

    for chunk in tl.static_range(CHUNKS):
        byte = ((gray >> (chunk * 8)) & 0xFF).to(tl.int32)
        part = tl.load(chunk_ptr + (dim * CHUNKS + chunk) * 256 + byte, mask=mask, other=0)
        result = result ^ part.to(tl.uint32, bitcast=True)

    if HAS_SCRAMBLE:
        result = result ^ tl.load(scramble_ptr + dim).to(tl.uint32, bitcast=True)

    u = tl.uint_to_uniform_float(result.to(tl.uint32, bitcast=True))
    tl.store(out_ptr + dim * points_per_dim + point, 1.0 - u, mask=mask)


@triton.jit
def _sobol32_normal_chunk_table_kernel(
    out_ptr,
    chunk_ptr,
    scramble_ptr,
    points_per_dim,
    offset,
    mean,
    stddev,
    CHUNKS: tl.constexpr,
    HAS_SCRAMBLE: tl.constexpr,
    LOGNORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dim = tl.program_id(1)
    point = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = point < points_per_dim
    gray = (point + offset) ^ ((point + offset) >> 1)
    result = tl.full((BLOCK,), 0, tl.uint32)

    for chunk in tl.static_range(CHUNKS):
        byte = ((gray >> (chunk * 8)) & 0xFF).to(tl.int32)
        part = tl.load(chunk_ptr + (dim * CHUNKS + chunk) * 256 + byte, mask=mask, other=0)
        result = result ^ part.to(tl.uint32, bitcast=True)

    if HAS_SCRAMBLE:
        result = result ^ tl.load(scramble_ptr + dim).to(tl.uint32, bitcast=True)

    u = 1.0 - tl.uint_to_uniform_float(result.to(tl.uint32, bitcast=True))
    y = mean + stddev * uniform_to_normal_icdf_f32(u)
    if LOGNORMAL:
        y = tl.exp(y)
    tl.store(out_ptr + dim * points_per_dim + point, y, mask=mask)


_SOBOL32_RAW_LAUNCHER = CachedKernelLauncher(
    _sobol32_chunk_table_kernel,
    constexpr_names=("CHUNKS", "HAS_SCRAMBLE", "BLOCK"),
)
_SOBOL32_UNIFORM_LAUNCHER = CachedKernelLauncher(
    _sobol32_uniform_chunk_table_kernel,
    constexpr_names=("CHUNKS", "HAS_SCRAMBLE", "BLOCK"),
)
_SOBOL32_NORMAL_LAUNCHER = CachedKernelLauncher(
    _sobol32_normal_chunk_table_kernel,
    constexpr_names=("CHUNKS", "HAS_SCRAMBLE", "LOGNORMAL", "BLOCK"),
)


def launch_sobol32_table(
    out: torch.Tensor,
    direction_vectors: torch.Tensor,
    *,
    dimensions: int,
    offset: int,
    block_size: int,
    num_warps: int,
    scramble_constants: torch.Tensor | None = None,
) -> None:
    points_per_dim = out.numel() // dimensions
    chunk_device, scramble_device, chunks, grid_x = launch_plan32(
        out, direction_vectors, scramble_constants, dimensions, points_per_dim, offset, block_size
    )
    grid = (grid_x, dimensions)
    _SOBOL32_RAW_LAUNCHER.launch(
        grid,
        (out, chunk_device, scramble_device, points_per_dim, offset),
        (chunks, scramble_constants is not None, block_size),
        (num_warps,),
    )


def launch_sobol32_uniform_table(
    out: torch.Tensor,
    direction_vectors: torch.Tensor,
    *,
    dimensions: int,
    offset: int,
    block_size: int,
    num_warps: int,
    scramble_constants: torch.Tensor | None = None,
) -> None:
    points_per_dim = out.numel() // dimensions
    chunk_device, scramble_device, chunks, grid_x = launch_plan32(
        out, direction_vectors, scramble_constants, dimensions, points_per_dim, offset, block_size
    )
    grid = (grid_x, dimensions)
    _SOBOL32_UNIFORM_LAUNCHER.launch(
        grid,
        (out, chunk_device, scramble_device, points_per_dim, offset),
        (chunks, scramble_constants is not None, block_size),
        (num_warps,),
    )


def launch_sobol32_normal_table(
    out: torch.Tensor,
    direction_vectors: torch.Tensor,
    *,
    dimensions: int,
    offset: int,
    mean: float,
    stddev: float,
    lognormal: bool,
    block_size: int,
    num_warps: int,
    scramble_constants: torch.Tensor | None = None,
) -> None:
    points_per_dim = out.numel() // dimensions
    chunk_device, scramble_device, chunks, grid_x = launch_plan32(
        out, direction_vectors, scramble_constants, dimensions, points_per_dim, offset, block_size
    )
    grid = (grid_x, dimensions)
    _SOBOL32_NORMAL_LAUNCHER.launch(
        grid,
        (out, chunk_device, scramble_device, points_per_dim, offset, mean, stddev),
        (chunks, scramble_constants is not None, lognormal, block_size),
        (num_warps,),
    )
