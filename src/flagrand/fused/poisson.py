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
def _poisson_transform_kernel_small_32(out_ptr, raw_ptr, n, lambda_val, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(raw_ptr + offs, mask=mask, other=0)
    u = uint32_to_uniform(x)
    k = tl.floor(-tl.log(tl.maximum(u, 1e-7)) * lambda_val + 0.5)
    tl.store(out_ptr + offs, k.to(tl.int32), mask=mask)


@triton.jit
def _poisson_transform_kernel_large_32(out_ptr, raw_ptr, n, lambda_val, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = (offs * 2)[:, None] + tl.arange(0, 2)[None, :]
    x = tl.load(raw_ptr + base, mask=mask[:, None], other=0)
    x0, x1 = tl.split(x)
    u0 = uint32_to_uniform(x0)
    u1 = uint32_to_uniform(x1)
    n0, n1 = uniform_to_normal(u0, u1)
    k0 = tl.maximum(0, tl.floor(lambda_val + tl.sqrt(lambda_val) * n0))
    k1 = tl.maximum(0, tl.floor(lambda_val + tl.sqrt(lambda_val) * n1))
    out = tl.join(k0.to(tl.int32), k1.to(tl.int32))
    tl.store(out_ptr + base, out, mask=mask[:, None])


@triton.jit
def _poisson_transform_kernel_small_64(out_ptr, raw_ptr, n, lambda_val, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(raw_ptr + offs, mask=mask, other=0)
    u = uint64_to_uniform64(x)
    k = tl.floor(-tl.log(tl.maximum(u, 1e-7)) * lambda_val + 0.5)
    tl.store(out_ptr + offs, k.to(tl.int64), mask=mask)


@triton.jit
def _poisson_transform_kernel_large_64(out_ptr, raw_ptr, n, lambda_val, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = (offs * 2)[:, None] + tl.arange(0, 2)[None, :]
    x = tl.load(raw_ptr + base, mask=mask[:, None], other=0)
    x0, x1 = tl.split(x)
    u0 = uint64_to_uniform64(x0)
    u1 = uint64_to_uniform64(x1)
    n0, n1 = uniform_to_normal(u0, u1)
    k0 = tl.maximum(0, tl.floor(lambda_val + tl.sqrt(lambda_val) * n0))
    k1 = tl.maximum(0, tl.floor(lambda_val + tl.sqrt(lambda_val) * n1))
    out = tl.join(k0.to(tl.int64), k1.to(tl.int64))
    tl.store(out_ptr + base, out, mask=mask[:, None])


def generate_poisson(
    out: torch.Tensor,
    generator,
    *,
    lambda_val: float,
    block_size: int = 512,
    num_warps: int = 4,
) -> torch.Tensor:
    require_accelerator()

    gen_type = get_generator_type(generator)
    is_64 = gen_type in _64BIT_GENERATORS

    if is_64:
        if out.dtype != torch.int64:
            raise TypeError("generate_poisson: int64 output required for Generator64.")
    else:
        if out.dtype != torch.int32:
            raise TypeError("generate_poisson: int32 output required for Generator32.")

    assert_tensor_device_supported(out, op_name="generate_poisson")

    n = out.numel()
    if n == 0:
        return out
    if lambda_val <= 0:
        raise ValueError(f"generate_poisson: lambda must be > 0, got {lambda_val}.")
    if block_size <= 0:
        raise ValueError(f"generate_poisson: block_size must be > 0, got {block_size}.")
    if num_warps <= 0:
        raise ValueError(f"generate_poisson: num_warps must be > 0, got {num_warps}.")

    if not is_64 and gen_type == GENERATOR_PHILOX:
        if n % 4 != 0:
            raise ValueError(
                f"generate_poisson: Philox requires element count to be "
                f"a multiple of 4, got {n}."
            )

    if is_64:
        raw = _generate_raw64(generator, out.shape, out.device)
        if lambda_val < 30.0:
            grid = (triton.cdiv(n, block_size),)
            _poisson_transform_kernel_small_64[grid](
                out.view(-1), raw.view(-1), n,
                lambda_val,
                BLOCK=block_size, num_warps=num_warps,
            )
        else:
            n_pairs = n // 2
            grid = (triton.cdiv(n_pairs, block_size),)
            _poisson_transform_kernel_large_64[grid](
                out.view(-1), raw.view(-1), n_pairs,
                lambda_val,
                BLOCK=block_size, num_warps=num_warps,
            )
    else:
        raw = _generate_raw(generator, out.shape, out.device)
        if lambda_val < 30.0:
            grid = (triton.cdiv(n, block_size),)
            _poisson_transform_kernel_small_32[grid](
                out.view(-1), raw.view(-1), n,
                lambda_val,
                BLOCK=block_size, num_warps=num_warps,
            )
        else:
            n_pairs = n // 2
            grid = (triton.cdiv(n_pairs, block_size),)
            _poisson_transform_kernel_large_32[grid](
                out.view(-1), raw.view(-1), n_pairs,
                lambda_val,
                BLOCK=block_size, num_warps=num_warps,
            )

    return out
