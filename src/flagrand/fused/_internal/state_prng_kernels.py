from __future__ import annotations

import triton
import triton.language as tl

from flagrand.fused._internal.state_prng_state import init_state, step_state
from flagrand.fused._internal.transforms import uint32_to_uniform, uniform_to_normal


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
def uniform_kernel(
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
    s0, s1, s2, s3, s4, s5 = init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)

    for k in range(num_iters):
        raw, s0, s1, s2, s3, s4, s5 = step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        out_offs = k * n_threads + tid
        out_mask = thread_mask & (out_offs < n)
        tl.store(out_ptr + out_offs, uint32_to_uniform(raw), mask=out_mask)


@triton.jit
def normal_kernel(
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
    s0, s1, s2, s3, s4, s5 = init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)

    for k in range(num_iters):
        raw0, s0, s1, s2, s3, s4, s5 = step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        raw1, s0, s1, s2, s3, s4, s5 = step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
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
def poisson_small_kernel(
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
    s0, s1, s2, s3, s4, s5 = init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)

    for k in range(num_iters):
        raw, s0, s1, s2, s3, s4, s5 = step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        out_offs = k * n_threads + tid
        out_mask = thread_mask & (out_offs < n)
        value = _poisson_inverse_from_uniform(uint32_to_uniform(raw), lambda_val, MAX_K)
        tl.store(out_ptr + out_offs, value, mask=out_mask)


@triton.jit
def poisson_large_kernel(
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
    s0, s1, s2, s3, s4, s5 = init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND)
    sigma = tl.sqrt(lambda_val)

    for k in range(num_iters):
        raw0, s0, s1, s2, s3, s4, s5 = step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        raw1, s0, s1, s2, s3, s4, s5 = step_state(s0, s1, s2, s3, s4, s5, RNG_KIND)
        n0, n1 = uniform_to_normal(uint32_to_uniform(raw0), uint32_to_uniform(raw1))

        k0 = tl.maximum(0, tl.floor(lambda_val + sigma * n0)).to(tl.int32)
        k1 = tl.maximum(0, tl.floor(lambda_val + sigma * n1)).to(tl.int32)

        pair = k * n_threads + tid
        base = pair * 2
        pair_mask = thread_mask & (pair < n_pairs)
        tl.store(out_ptr + base + 0, k0, mask=pair_mask)
        tl.store(out_ptr + base + 1, k1, mask=pair_mask)
