from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


BOX_MULLER_MIN_U_F32 = 2.3283064365386963e-10
BOX_MULLER_MIN_U_F64 = 1.1102230246251565e-16
_BOX_MULLER_MIN_U_F32_TL = tl.constexpr(BOX_MULLER_MIN_U_F32)
_BOX_MULLER_MIN_U_F64_TL = tl.constexpr(BOX_MULLER_MIN_U_F64)


@triton.jit
def uint32_to_uniform(x):
    return uint32_to_uniform_curand_compat(x)


@triton.jit
def uint32_to_uniform_curand_compat(x):
    u = tl.uint_to_uniform_float(x.to(tl.uint32, bitcast=True))
    return 1.0 - u


@triton.jit
def uint32_to_uniform_native(x):
    x = x.to(tl.uint32, bitcast=True)
    mantissa = x >> 8
    return mantissa.to(tl.float32) * 5.960464477539063e-8


@triton.jit
def uint64_to_uniform64(x):
    return uint64_to_uniform64_curand_compat(x)


@triton.jit
def uint64_to_uniform64_curand_compat(x):
    u = uint64_to_uniform64_native(x)
    return 1.0 - u


@triton.jit
def uint64_to_uniform64_native(x):
    x = x.to(tl.uint64, bitcast=True)
    return (x >> 11).to(tl.float64) * 1.1102230246251565e-16


@triton.jit
def uint32_pair_to_uniform64_curand_compat(x0, x1):
    native = uint32_pair_to_uniform64_native(x0, x1)
    return 1.0 - native


@triton.jit
def uint32_pair_to_uniform64_native(x0, x1):
    hi = (x0.to(tl.uint32, bitcast=True) >> 5).to(tl.uint64)
    lo = (x1.to(tl.uint32, bitcast=True) >> 6).to(tl.uint64)
    mantissa = (hi << 26) | lo
    return mantissa.to(tl.float64) * 1.1102230246251565e-16


@triton.jit
def uniform_to_normal(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F32_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta_pi = 2.0 * u2
    n1 = r * libdevice.cospi(theta_pi)
    n2 = r * libdevice.sinpi(theta_pi)
    return n1, n2


@triton.jit
def uniform_to_normal_trig(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F32_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = 6.283185307179586 * u2
    n1 = r * tl.cos(theta)
    n2 = r * tl.sin(theta)
    return n1, n2


@triton.jit
def uniform_to_normal_fast_f64(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F64_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    sine, cosine = _sincos_poly_f64(u2)
    n1 = r * cosine
    n2 = r * sine
    return n1, n2


@triton.jit
def uniform_to_normal_fast_f32(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F32_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = 6.283185307179586 * u2
    n1 = r * _cos_approx_f32(theta)
    n2 = r * _sin_approx_f32(theta)
    return n1, n2


@triton.jit
def uniform_to_normal_icdf_f32(u):
    u = tl.where(u <= 0.0, 5.960464477539063e-8, u)
    u = tl.where(u >= 1.0, 0.9999999403953552, u)
    return 1.4142135623730951 * libdevice.erfinv(2.0 * u - 1.0)


@triton.jit
def uniform_to_normal_icdf_f64(u):
    u = tl.where(u <= 0.0, 1.0e-12, u)
    u = tl.where(u >= 1.0, 0.999999, u)
    return 1.4142135623730951 * libdevice.erfinv(2.0 * u - 1.0)

@triton.jit
def _sin_approx_f32(x):
    return tl.inline_asm_elementwise(
        "sin.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cos_approx_f32(x):
    return tl.inline_asm_elementwise(
        "cos.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sincos_poly_f64(unit_angle):
    quadrant = tl.floor(4.0 * unit_angle + 0.5).to(tl.int32)
    reduced = (
        6.283185307179586476925286766559 * unit_angle
        - 1.5707963267948966192313216916398 * quadrant.to(tl.float64)
    )
    squared = reduced * reduced

    sine_coeff = 1.0 / 355687428096000.0
    sine_coeff = -1.0 / 1307674368000.0 + squared * sine_coeff
    sine_coeff = 1.0 / 6227020800.0 + squared * sine_coeff
    sine_coeff = -1.0 / 39916800.0 + squared * sine_coeff
    sine_coeff = 1.0 / 362880.0 + squared * sine_coeff
    sine_coeff = -1.0 / 5040.0 + squared * sine_coeff
    sine_coeff = 1.0 / 120.0 + squared * sine_coeff
    sine_coeff = -1.0 / 6.0 + squared * sine_coeff
    sine = reduced + reduced * squared * sine_coeff

    cosine_coeff = 1.0 / 20922789888000.0
    cosine_coeff = -1.0 / 87178291200.0 + squared * cosine_coeff
    cosine_coeff = 1.0 / 479001600.0 + squared * cosine_coeff
    cosine_coeff = -1.0 / 3628800.0 + squared * cosine_coeff
    cosine_coeff = 1.0 / 40320.0 + squared * cosine_coeff
    cosine_coeff = -1.0 / 720.0 + squared * cosine_coeff
    cosine_coeff = 1.0 / 24.0 + squared * cosine_coeff
    cosine_coeff = -0.5 + squared * cosine_coeff
    cosine = 1.0 + squared * cosine_coeff

    quadrant = quadrant & 3
    restored_sine = tl.where(
        quadrant == 0,
        sine,
        tl.where(quadrant == 1, cosine, tl.where(quadrant == 2, -sine, -cosine)),
    )
    restored_cosine = tl.where(
        quadrant == 0,
        cosine,
        tl.where(quadrant == 1, -sine, tl.where(quadrant == 2, -cosine, sine)),
    )
    return restored_sine, restored_cosine
