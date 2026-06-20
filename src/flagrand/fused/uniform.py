from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand._device import require_accelerator, assert_tensor_device_supported
from flagrand.fused._internal.transforms import uint32_to_uniform, uint64_to_uniform64
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
def _uniform_transform_kernel_32(out_ptr, raw_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(raw_ptr + offs, mask=mask, other=0)
    u = uint32_to_uniform(x)
    tl.store(out_ptr + offs, u, mask=mask)


@triton.jit
def _uniform_transform_kernel_64(out_ptr, raw_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(raw_ptr + offs, mask=mask, other=0)
    u = uint64_to_uniform64(x)
    tl.store(out_ptr + offs, u, mask=mask)


def generate_uniform(
    out: torch.Tensor,
    generator,
    *,
    block_size: int = 512,
    num_warps: int = 4,
) -> torch.Tensor:
    require_accelerator()

    gen_type = get_generator_type(generator)
    is_64 = gen_type in _64BIT_GENERATORS

    if is_64:
        if out.dtype != torch.float64:
            raise TypeError("generate_uniform: float64 output required for Generator64.")
    else:
        if out.dtype != torch.float32:
            raise TypeError("generate_uniform: float32 output required for Generator32.")

    assert_tensor_device_supported(out, op_name="generate_uniform")

    n = out.numel()
    if n == 0:
        return out
    if block_size <= 0:
        raise ValueError(f"generate_uniform: block_size must be > 0, got {block_size}.")
    if num_warps <= 0:
        raise ValueError(f"generate_uniform: num_warps must be > 0, got {num_warps}.")

    if not is_64 and gen_type == GENERATOR_PHILOX:
        if n % 4 != 0:
            raise ValueError(
                f"generate_uniform: Philox requires element count to be "
                f"a multiple of 4, got {n}."
            )

    if is_64:
        raw = _generate_raw64(generator, out.shape, out.device)
        grid = (triton.cdiv(n, block_size),)
        _uniform_transform_kernel_64[grid](
            out.view(-1), raw.view(-1), n,
            BLOCK=block_size, num_warps=num_warps,
        )
    else:
        raw = _generate_raw(generator, out.shape, out.device)
        grid = (triton.cdiv(n, block_size),)
        _uniform_transform_kernel_32[grid](
            out.view(-1), raw.view(-1), n,
            BLOCK=block_size, num_warps=num_warps,
        )

    return out
