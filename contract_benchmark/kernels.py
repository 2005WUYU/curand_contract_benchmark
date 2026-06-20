from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pure_write_f32_kernel(out_ptr, n, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, value + offs.to(tl.float32) * 0.0, mask=mask)


@triton.jit
def _consume_add_uniform_kernel(x_ptr, u_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0)
    y = x + alpha * (u - 0.5)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _threshold_from_uniform_kernel(u_ptr, mask_ptr, n, p, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    u = tl.load(u_ptr + offs, mask=mask, other=1.0)
    keep = u <= p
    tl.store(mask_ptr + offs, keep.to(tl.uint8), mask=mask)


@triton.jit
def _dropout_from_uniform_kernel(x_ptr, u_ptr, out_ptr, mask_ptr, n, p, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    u = tl.load(u_ptr + offs, mask=mask, other=1.0)
    keep = u <= p
    y = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)
    tl.store(mask_ptr + offs, keep.to(tl.uint8), mask=mask)


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _fused_philox_add_uniform_kernel(x_ptr, out_ptr, seed, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    counters = pid * BLOCK + tl.arange(0, BLOCK)
    r0, r1, r2, r3 = _philox_generate(seed, counters)
    r01 = tl.join(r0, r1)
    r23 = tl.join(r2, r3)
    tile = tl.reshape(tl.join(r01, r23), (BLOCK, 4))
    base = (counters * 4)[:, None] + tl.arange(0, 4)[None, :]
    mask = base < n
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    u = tl.uint_to_uniform_float(tile.to(tl.uint32, bitcast=True))
    y = x + alpha * (u - 0.5)
    tl.store(out_ptr + base, y, mask=mask)


@triton.jit
def _fused_philox_threshold_kernel(mask_ptr, seed, n, p, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    counters = pid * BLOCK + tl.arange(0, BLOCK)
    r0, r1, r2, r3 = _philox_generate(seed, counters)
    r01 = tl.join(r0, r1)
    r23 = tl.join(r2, r3)
    tile = tl.reshape(tl.join(r01, r23), (BLOCK, 4))
    base = (counters * 4)[:, None] + tl.arange(0, 4)[None, :]
    mask = base < n
    u = tl.uint_to_uniform_float(tile.to(tl.uint32, bitcast=True))
    keep = u <= p
    tl.store(mask_ptr + base, keep.to(tl.uint8), mask=mask)


@triton.jit
def _fused_philox_dropout_kernel(x_ptr, out_ptr, mask_ptr, seed, n, p, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    counters = pid * BLOCK + tl.arange(0, BLOCK)
    r0, r1, r2, r3 = _philox_generate(seed, counters)
    r01 = tl.join(r0, r1)
    r23 = tl.join(r2, r3)
    tile = tl.reshape(tl.join(r01, r23), (BLOCK, 4))
    base = (counters * 4)[:, None] + tl.arange(0, 4)[None, :]
    mask = base < n
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    u = tl.uint_to_uniform_float(tile.to(tl.uint32, bitcast=True))
    keep = u <= p
    y = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + base, y, mask=mask)
    tl.store(mask_ptr + base, keep.to(tl.uint8), mask=mask)


def pure_write_f32(out: torch.Tensor, *, value: float = 0.0, block_size: int = 256) -> torch.Tensor:
    n = out.numel()
    grid = (triton.cdiv(n, block_size),)
    _pure_write_f32_kernel[grid](out, n, float(value), BLOCK=block_size, num_warps=4)
    return out


def consume_add_uniform(
    x: torch.Tensor,
    uniform: torch.Tensor,
    out: torch.Tensor,
    *,
    alpha: float,
    block_size: int = 256,
) -> torch.Tensor:
    n = out.numel()
    grid = (triton.cdiv(n, block_size),)
    _consume_add_uniform_kernel[grid](x, uniform, out, n, float(alpha), BLOCK=block_size, num_warps=4)
    return out


def threshold_from_uniform(
    uniform: torch.Tensor,
    mask: torch.Tensor,
    *,
    p: float,
    block_size: int = 256,
) -> torch.Tensor:
    n = mask.numel()
    grid = (triton.cdiv(n, block_size),)
    _threshold_from_uniform_kernel[grid](uniform, mask, n, float(p), BLOCK=block_size, num_warps=4)
    return mask


def dropout_from_uniform(
    x: torch.Tensor,
    uniform: torch.Tensor,
    out: torch.Tensor,
    mask: torch.Tensor,
    *,
    p: float,
    block_size: int = 256,
) -> torch.Tensor:
    n = out.numel()
    scale = 1.0 / float(p)
    grid = (triton.cdiv(n, block_size),)
    _dropout_from_uniform_kernel[grid](x, uniform, out, mask, n, float(p), scale, BLOCK=block_size, num_warps=4)
    return out


def fused_philox_add_uniform(
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    seed: int,
    alpha: float,
    block_size: int = 256,
) -> torch.Tensor:
    n = out.numel()
    _require_multiple_of_4(n, "fused_philox_add_uniform")
    grid = (triton.cdiv(n // 4, block_size),)
    _fused_philox_add_uniform_kernel[grid](x, out, int(seed), n, float(alpha), BLOCK=block_size, num_warps=4)
    return out


def fused_philox_threshold(
    mask: torch.Tensor,
    *,
    seed: int,
    p: float,
    block_size: int = 256,
) -> torch.Tensor:
    n = mask.numel()
    _require_multiple_of_4(n, "fused_philox_threshold")
    grid = (triton.cdiv(n // 4, block_size),)
    _fused_philox_threshold_kernel[grid](mask, int(seed), n, float(p), BLOCK=block_size, num_warps=4)
    return mask


def fused_philox_dropout(
    x: torch.Tensor,
    out: torch.Tensor,
    mask: torch.Tensor,
    *,
    seed: int,
    p: float,
    block_size: int = 256,
) -> torch.Tensor:
    n = out.numel()
    _require_multiple_of_4(n, "fused_philox_dropout")
    scale = 1.0 / float(p)
    grid = (triton.cdiv(n // 4, block_size),)
    _fused_philox_dropout_kernel[grid](x, out, mask, int(seed), n, float(p), scale, BLOCK=block_size, num_warps=4)
    return out


def _require_multiple_of_4(n: int, op_name: str) -> None:
    if n % 4 != 0:
        raise ValueError(f"{op_name} requires N % 4 == 0 for Philox4 lanes, got {n}")

