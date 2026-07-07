from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand._device import require_accelerator, assert_tensor_device_supported
from flagrand.fused._internal.transforms import uint32_to_uniform, uint64_to_uniform64
from flagrand.fused._internal.utils import (
    get_generator_type,
    GENERATOR_PHILOX,
    GENERATOR_MTGP32,
    GENERATOR_SOBOL64,
    GENERATOR_SCRAMBLED_SOBOL64,
    _generate_raw,
    _generate_raw64,
)

_64BIT_GENERATORS = {GENERATOR_SOBOL64, GENERATOR_SCRAMBLED_SOBOL64}


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _philox_uniform_kernel(out_ptr, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    counter = base_counter + offs
    r0, r1, r2, r3 = _philox_generate(seed, counter)
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)
    u01 = tl.join(u0, u1)
    u23 = tl.join(u2, u3)
    tile = tl.reshape(tl.join(u01, u23), (BLOCK, 4))
    base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + base, tile, mask=mask[:, None])


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
        _generate_philox_uniform(out, generator, block_size, num_warps)
        return out
    if not is_64 and gen_type == GENERATOR_MTGP32:
        generator.generate_uniform(out)
        return out

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


def _generate_philox_uniform(
    out: torch.Tensor,
    generator,
    block_size: int,
    num_warps: int,
) -> None:
    n = out.numel()
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"generate_uniform: Philox offset must be >= 0, got {offset_val}.")
    if offset_val % 4 != 0:
        raise ValueError(
            "generate_uniform: Philox offset is measured in uint32 outputs "
            f"and must be a multiple of 4, got {offset_val}."
        )
    seed_val = int(getattr(generator, "seed", 0))
    n_counters = n // 4
    grid = (triton.cdiv(n_counters, block_size),)
    _philox_uniform_kernel[grid](
        out.view(-1),
        seed_val,
        offset_val // 4,
        n_counters,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    generator.offset = offset_val + n
