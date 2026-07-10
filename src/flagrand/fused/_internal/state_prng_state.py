from __future__ import annotations

import triton
import triton.language as tl

RNG_XORWOW: int = 0
RNG_MRG32K3A: int = 1


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
    return (
        v0.to(tl.int64),
        v1.to(tl.int64),
        v2.to(tl.int64),
        v3.to(tl.int64),
        v4.to(tl.int64),
        d.to(tl.int64),
    )


@triton.jit
def _xorwow_step(v0, v1, v2, v3, v4, d):
    v0 = v0.to(tl.uint32)
    v1 = v1.to(tl.uint32)
    v2 = v2.to(tl.uint32)
    v3 = v3.to(tl.uint32)
    v4 = v4.to(tl.uint32)
    d = d.to(tl.uint32)
    t = v0 ^ (v0 >> 2)
    nv0, nv1, nv2, nv3 = v1, v2, v3, v4
    nv4 = (v4 ^ (v4 << 4)) ^ (t ^ (t << 1))
    nd = d + 362437
    return (
        (nv4 + nd).to(tl.uint32),
        nv0.to(tl.int64),
        nv1.to(tl.int64),
        nv2.to(tl.int64),
        nv3.to(tl.int64),
        nv4.to(tl.int64),
        nd.to(tl.int64),
    )


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
def init_state(seed_lo, seed_hi, tid, offset_u32, RNG_KIND: tl.constexpr):
    if RNG_KIND == 0:
        return _xorwow_init(seed_lo, seed_hi, tid, offset_u32)
    return _mrg32k3a_init(seed_lo, tid, offset_u32)


@triton.jit
def step_state(s0, s1, s2, s3, s4, s5, RNG_KIND: tl.constexpr):
    if RNG_KIND == 0:
        return _xorwow_step(s0, s1, s2, s3, s4, s5)
    return _mrg32k3a_step(s0, s1, s2, s3, s4, s5)
