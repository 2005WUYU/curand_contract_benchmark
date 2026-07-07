from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.transforms import uint32_to_uniform, uniform_to_normal

_RNG_XORWOW: int = 0
_RNG_MRG32K3A: int = 1
_BLOCK: int = 128
_TARGET_THREADS: int = 131072


@triton.jit
def _splitmix32(x):
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _xorwow_init(seed_lo, seed_hi, tid, offset_u32):
    tid_u = tid.to(tl.uint32)
    sl = seed_lo ^ _splitmix32(tid_u + offset_u32)
    sh = seed_hi ^ _splitmix32(tid_u + offset_u32 + 0x9E3779B9)

    t0 = 1099087573 * sl
    t1 = 2591861531 * sh

    d = 6615241 + t1 + t0
    v0 = 123456789 + t0
    v1 = 362436069 ^ t0
    v2 = 521288629 + t1
    v3 = 88675123 ^ t1
    v4 = 5783321 + t0
    return v0, v1, v2, v3, v4, d


@triton.jit
def _xorwow_step(v0, v1, v2, v3, v4, d):
    t = v0 ^ (v0 >> 2)
    nv0, nv1, nv2, nv3 = v1, v2, v3, v4
    nv4 = (v4 ^ (v4 << 4)) ^ (t ^ (t << 1))
    nd = d + 362437
    return (nv4 + nd).to(tl.uint32), nv0, nv1, nv2, nv3, nv4, nd


@triton.jit
def _mrg32k3a_init(seed_u32, tid, offset_u32):
    tid_u = tid.to(tl.uint32)
    pert = _splitmix32(seed_u32 + tid_u + offset_u32)
    pert_b = _splitmix32(seed_u32 + tid_u + offset_u32 + 0x9E3779B9)

    pert64 = pert.to(tl.int64)
    pert_b64 = pert_b.to(tl.int64)

    s1_0 = (123456789 + pert64) % 4294967087
    s1_1 = (362436069 + pert_b64) % 4294967087
    s1_2 = (521288629 + pert64 + pert_b64) % 4294967087
    s2_0 = (88675123 + pert64) % 4294944443
    s2_1 = (5783321 + pert_b64) % 4294944443
    s2_2 = (6615241 + pert64 + pert_b64) % 4294944443

    s1_0 = tl.where(s1_0 == 0, 1, s1_0)
    s1_1 = tl.where(s1_1 == 0, 1, s1_1)
    s1_2 = tl.where(s1_2 == 0, 1, s1_2)
    s2_0 = tl.where(s2_0 == 0, 1, s2_0)
    s2_1 = tl.where(s2_1 == 0, 1, s2_1)
    s2_2 = tl.where(s2_2 == 0, 1, s2_2)
    return s1_0, s1_1, s1_2, s2_0, s2_1, s2_2


@triton.jit
def _mrg32k3a_step(s1_0, s1_1, s1_2, s2_0, s2_1, s2_2):
    x1 = (1403580 * s1_1 + 4294156359 * s1_2) % 4294967087
    x2 = (527612 * s2_0 + 4293573854 * s2_2) % 4294944443
    diff = x1 - x2
    output = tl.where(diff < 0, diff + 4294967087, diff)
    output_u32 = (output & 0xFFFFFFFF).to(tl.uint32)
    return output_u32, s1_1, s1_2, x1, s2_1, s2_2, x2


@triton.jit
def _init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND: tl.constexpr):
    if RNG_KIND == 0:
        return _xorwow_init(seed_lo, seed_hi, tid, offset_u32)
    return _mrg32k3a_init(seed_lo, tid, offset_u32)


@triton.jit
def _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND: tl.constexpr):
    if RNG_KIND == 0:
        return _xorwow_step(s0, s1, s2, s3, s4, s5)
    return _mrg32k3a_step(s0, s1, s2, s3, s4, s5)


@triton.jit
def _poisson_inverse_from_uniform(u, lambda_val, MAX_K: tl.constexpr):
    p = tl.exp(-lambda_val)
    cdf = p
    k = tl.full(u.shape, 0, tl.int32)
    for i in range(1, MAX_K + 1):
        active = u > cdf
        p = p * lambda_val / i
        cdf += p
        k = tl.where(active, i, k)
    return k


@triton.jit
def _uniform_kernel(
    out_ptr,
    seed_lo,
    seed_hi,
    offset_u32,
    n,
    n_threads,
    num_iters,
    BLOCK: tl.constexpr,
    RNG_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    thread_mask = tid < n_threads
    s0, s1, s2, s3, s4, s5 = _init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)

    for k in range(num_iters):
        raw, s0, s1, s2, s3, s4, s5 = _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        out_offs = k * n_threads + tid
        out_mask = thread_mask & (out_offs < n)
        tl.store(out_ptr + out_offs, uint32_to_uniform(raw), mask=out_mask)


@triton.jit
def _normal_kernel(
    out_ptr,
    seed_lo,
    seed_hi,
    offset_u32,
    n_pairs,
    n_threads,
    num_iters,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
    RNG_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    thread_mask = tid < n_threads
    s0, s1, s2, s3, s4, s5 = _init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)

    for k in range(num_iters):
        raw0, s0, s1, s2, s3, s4, s5 = _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        raw1, s0, s1, s2, s3, s4, s5 = _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        n0, n1 = uniform_to_normal(uint32_to_uniform(raw0), uint32_to_uniform(raw1))

        y0 = mean + stddev * n0
        y1 = mean + stddev * n1
        if LOGNORMAL:
            y0 = tl.exp(y0)
            y1 = tl.exp(y1)

        pair = k * n_threads + tid
        base = pair * 2
        pair_mask = thread_mask & (pair < n_pairs)
        tl.store(out_ptr + base + 0, y0, mask=pair_mask)
        tl.store(out_ptr + base + 1, y1, mask=pair_mask)


@triton.jit
def _poisson_small_kernel(
    out_ptr,
    seed_lo,
    seed_hi,
    offset_u32,
    n,
    n_threads,
    num_iters,
    lambda_val,
    BLOCK: tl.constexpr,
    MAX_K: tl.constexpr,
    RNG_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    thread_mask = tid < n_threads
    s0, s1, s2, s3, s4, s5 = _init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)

    for k in range(num_iters):
        raw, s0, s1, s2, s3, s4, s5 = _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        out_offs = k * n_threads + tid
        out_mask = thread_mask & (out_offs < n)
        value = _poisson_inverse_from_uniform(uint32_to_uniform(raw), lambda_val, MAX_K)
        tl.store(out_ptr + out_offs, value, mask=out_mask)


@triton.jit
def _poisson_large_kernel(
    out_ptr,
    seed_lo,
    seed_hi,
    offset_u32,
    n_pairs,
    n_threads,
    num_iters,
    lambda_val,
    BLOCK: tl.constexpr,
    RNG_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    thread_mask = tid < n_threads
    s0, s1, s2, s3, s4, s5 = _init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)
    sigma = tl.sqrt(lambda_val)

    for k in range(num_iters):
        raw0, s0, s1, s2, s3, s4, s5 = _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        raw1, s0, s1, s2, s3, s4, s5 = _step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        n0, n1 = uniform_to_normal(uint32_to_uniform(raw0), uint32_to_uniform(raw1))

        k0 = tl.maximum(0, tl.floor(lambda_val + sigma * n0)).to(tl.int32)
        k1 = tl.maximum(0, tl.floor(lambda_val + sigma * n1)).to(tl.int32)

        pair = k * n_threads + tid
        base = pair * 2
        pair_mask = thread_mask & (pair < n_pairs)
        tl.store(out_ptr + base + 0, k0, mask=pair_mask)
        tl.store(out_ptr + base + 1, k1, mask=pair_mask)


def generate_xorwow_uniform_f32(out: torch.Tensor, generator) -> None:
    _launch_uniform(out, generator, _RNG_XORWOW, "generate_uniform")


def generate_xorwow_normal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, _RNG_XORWOW, mean, stddev, lognormal=False, op_name="generate_normal")


def generate_xorwow_lognormal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, _RNG_XORWOW, mean, stddev, lognormal=True, op_name="generate_lognormal")


def generate_xorwow_poisson_u32(out: torch.Tensor, generator, *, lambda_val: float, max_k: int) -> None:
    _launch_poisson(out, generator, _RNG_XORWOW, lambda_val, max_k, "generate_poisson")


def generate_mrg32k3a_uniform_f32(out: torch.Tensor, generator) -> None:
    _launch_uniform(out, generator, _RNG_MRG32K3A, "generate_uniform")


def generate_mrg32k3a_normal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, _RNG_MRG32K3A, mean, stddev, lognormal=False, op_name="generate_normal")


def generate_mrg32k3a_lognormal_f32(out: torch.Tensor, generator, *, mean: float, stddev: float) -> None:
    _launch_normal(out, generator, _RNG_MRG32K3A, mean, stddev, lognormal=True, op_name="generate_lognormal")


def generate_mrg32k3a_poisson_u32(out: torch.Tensor, generator, *, lambda_val: float, max_k: int) -> None:
    _launch_poisson(out, generator, _RNG_MRG32K3A, lambda_val, max_k, "generate_poisson")


def _launch_uniform(out: torch.Tensor, generator, rng_kind: int, op_name: str) -> None:
    seed_lo, seed_hi, offset_val = _launch_seed_args(generator, op_name)
    n = out.numel()
    n_threads, num_iters = _thread_plan(n)
    grid = (triton.cdiv(n_threads, _BLOCK),)
    _uniform_kernel[grid](
        out.view(-1),
        seed_lo,
        seed_hi,
        offset_val & 0xFFFFFFFF,
        n,
        n_threads,
        num_iters,
        BLOCK=_BLOCK,
        RNG_KIND=rng_kind,
        num_warps=4,
    )
    generator.offset = offset_val + n


def _launch_normal(
    out: torch.Tensor,
    generator,
    rng_kind: int,
    mean: float,
    stddev: float,
    *,
    lognormal: bool,
    op_name: str,
) -> None:
    seed_lo, seed_hi, offset_val = _launch_seed_args(generator, op_name)
    n_pairs = out.numel() // 2
    n_threads, num_iters = _thread_plan(n_pairs)
    grid = (triton.cdiv(n_threads, _BLOCK),)
    _normal_kernel[grid](
        out.view(-1),
        seed_lo,
        seed_hi,
        offset_val & 0xFFFFFFFF,
        n_pairs,
        n_threads,
        num_iters,
        mean,
        stddev,
        LOGNORMAL=lognormal,
        BLOCK=_BLOCK,
        RNG_KIND=rng_kind,
        num_warps=4,
    )
    generator.offset = offset_val + out.numel()


def _launch_poisson(
    out: torch.Tensor,
    generator,
    rng_kind: int,
    lambda_val: float,
    max_k: int,
    op_name: str,
) -> None:
    seed_lo, seed_hi, offset_val = _launch_seed_args(generator, op_name)
    if lambda_val < 30.0:
        n = out.numel()
        n_threads, num_iters = _thread_plan(n)
        grid = (triton.cdiv(n_threads, _BLOCK),)
        _poisson_small_kernel[grid](
            out.view(-1),
            seed_lo,
            seed_hi,
            offset_val & 0xFFFFFFFF,
            n,
            n_threads,
            num_iters,
            lambda_val,
            BLOCK=_BLOCK,
            MAX_K=max_k,
            RNG_KIND=rng_kind,
            num_warps=4,
        )
    else:
        n_pairs = out.numel() // 2
        n_threads, num_iters = _thread_plan(n_pairs)
        grid = (triton.cdiv(n_threads, _BLOCK),)
        _poisson_large_kernel[grid](
            out.view(-1),
            seed_lo,
            seed_hi,
            offset_val & 0xFFFFFFFF,
            n_pairs,
            n_threads,
            num_iters,
            lambda_val,
            BLOCK=_BLOCK,
            RNG_KIND=rng_kind,
            num_warps=4,
        )
    generator.offset = offset_val + out.numel()


def _thread_plan(n_work_items: int) -> tuple[int, int]:
    n_threads = min(_TARGET_THREADS, triton.cdiv(n_work_items, _BLOCK) * _BLOCK)
    n_threads = max(n_threads, _BLOCK)
    num_iters = triton.cdiv(n_work_items, n_threads)
    return n_threads, num_iters


def _launch_seed_args(generator, op_name: str) -> tuple[int, int, int]:
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"{op_name}: offset must be >= 0, got {offset_val}.")
    seed_val = int(getattr(generator, "seed", 0))
    return seed_val & 0xFFFFFFFF, (seed_val >> 32) & 0xFFFFFFFF, offset_val
