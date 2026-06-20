from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand._device import require_accelerator, assert_tensor_device_supported
from flagrand.fused._internal.transforms import uint32_to_uniform, uint64_to_uniform64, uniform_to_normal
from flagrand.fused._internal.utils import (
    get_generator_type,
    GENERATOR_PHILOX,
    GENERATOR_SOBOL64,
    GENERATOR_SCRAMBLED_SOBOL64,
    _generate_raw,
    _generate_raw64,
)

_64BIT_GENERATORS = {GENERATOR_SOBOL64, GENERATOR_SCRAMBLED_SOBOL64}


@triton.jit
def _normal_transform_kernel_32(out_ptr, raw_ptr, n, mean, stddev, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # 2D-tile load/store: (BLOCK, 2) lets the compiler emit ld.v2.b32 / st.v2.b32
    base = (offs * 2)[:, None] + tl.arange(0, 2)[None, :]
    x = tl.load(raw_ptr + base, mask=mask[:, None], other=0)
    x0, x1 = tl.split(x)
    u0 = uint32_to_uniform(x0)
    u1 = uint32_to_uniform(x1)
    n0, n1 = uniform_to_normal(u0, u1)
    n0 = mean + stddev * n0
    n1 = mean + stddev * n1
    out = tl.join(n0, n1)
    tl.store(out_ptr + base, out, mask=mask[:, None])


@triton.jit
def _normal_transform_kernel_64(out_ptr, raw_ptr, n, mean, stddev, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = (offs * 2)[:, None] + tl.arange(0, 2)[None, :]
    x = tl.load(raw_ptr + base, mask=mask[:, None], other=0)
    x0, x1 = tl.split(x)
    u0 = uint64_to_uniform64(x0)
    u1 = uint64_to_uniform64(x1)
    n0, n1 = uniform_to_normal(u0, u1)
    n0 = mean + stddev * n0
    n1 = mean + stddev * n1
    out = tl.join(n0, n1)
    tl.store(out_ptr + base, out, mask=mask[:, None])


def generate_normal(
    out: torch.Tensor,
    generator,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
    block_size: int = 512,
    num_warps: int = 4,
) -> torch.Tensor:
    require_accelerator()

    gen_type = get_generator_type(generator)
    is_64 = gen_type in _64BIT_GENERATORS

    if is_64:
        if out.dtype != torch.float64:
            raise TypeError("generate_normal: float64 output required for Generator64.")
    else:
        if out.dtype != torch.float32:
            raise TypeError("generate_normal: float32 output required for Generator32.")

    assert_tensor_device_supported(out, op_name="generate_normal")

    n = out.numel()
    if n == 0:
        return out
    if block_size <= 0:
        raise ValueError(f"generate_normal: block_size must be > 0, got {block_size}.")
    if num_warps <= 0:
        raise ValueError(f"generate_normal: num_warps must be > 0, got {num_warps}.")

    if not is_64 and gen_type == GENERATOR_PHILOX:
        if n % 4 != 0:
            raise ValueError(
                f"generate_normal: Philox requires element count to be "
                f"a multiple of 4, got {n}."
            )

    if is_64:
        raw = _generate_raw64(generator, out.shape, out.device)
        n_pairs = n // 2
        grid = (triton.cdiv(n_pairs, block_size),)
        _normal_transform_kernel_64[grid](
            out.view(-1), raw.view(-1), n_pairs,
            mean, stddev,
            BLOCK=block_size, num_warps=num_warps,
        )
    else:
        raw = _generate_raw(generator, out.shape, out.device)
        n_pairs = n // 2
        grid = (triton.cdiv(n_pairs, block_size),)
        _normal_transform_kernel_32[grid](
            out.view(-1), raw.view(-1), n_pairs,
            mean, stddev,
            BLOCK=block_size, num_warps=num_warps,
        )

    return out
