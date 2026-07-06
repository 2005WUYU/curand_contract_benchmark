from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def uint32_to_uniform(x):
    u = tl.uint_to_uniform_float(x.to(tl.uint32, bitcast=True))
    return tl.maximum(u, 2.3283064365386963e-10)


@triton.jit
def uint64_to_uniform64(x):
    # Use upper 53 bits of uint64 as mantissa for uniform float64
    x = x.to(tl.uint64, bitcast=True)
    u = (x >> 11).to(tl.float64) * (1.0 / (2**53))
    return tl.maximum(u, 1.1102230246251565e-16)


@triton.jit
def uniform_to_normal(u1, u2):
    u1 = tl.maximum(u1, 1e-7)
    u2 = tl.maximum(u2, 1e-7)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = 6.283185307179586 * u2
    n1 = r * tl.cos(theta)
    n2 = r * tl.sin(theta)
    return n1, n2
