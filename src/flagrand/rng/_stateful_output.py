from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import triton
import triton.language as tl

from flagrand.fused._internal.philox_poisson_table import _poisson_table_lookup
from flagrand.fused._internal.transforms import (
    uint32_to_uniform,
    uniform_to_normal_fast_f32,
)


OUTPUT_RAW = 0
OUTPUT_UNIFORM = 1
OUTPUT_NORMAL = 2
OUTPUT_LOGNORMAL = 3
OUTPUT_POISSON_SMALL = 4
OUTPUT_POISSON_LARGE = 5


@dataclass(frozen=True)
class StatefulOutput:
    mode: int = OUTPUT_RAW
    mean: float = 0.0
    stddev: float = 1.0
    lambda_val: float = 1.0
    max_k: int = 0

    @property
    def cache_key(self) -> tuple[int, float, float, float, int]:
        return (self.mode, self.mean, self.stddev, self.lambda_val, self.max_k)


RAW_OUTPUT = StatefulOutput()
UNIFORM_OUTPUT = StatefulOutput(mode=OUTPUT_UNIFORM)


@lru_cache(maxsize=64)
def normal_output(mean: float, stddev: float, *, lognormal: bool) -> StatefulOutput:
    return StatefulOutput(
        mode=OUTPUT_LOGNORMAL if lognormal else OUTPUT_NORMAL,
        mean=float(mean),
        stddev=float(stddev),
    )


@lru_cache(maxsize=64)
def poisson_output(lambda_val: float, max_k: int) -> StatefulOutput:
    return StatefulOutput(
        mode=OUTPUT_POISSON_SMALL if lambda_val < 30.0 else OUTPUT_POISSON_LARGE,
        lambda_val=float(lambda_val),
        max_k=int(max_k),
    )


def small_poisson_max_k(lambda_val: float) -> int:
    if lambda_val <= 0.1:
        return 16
    if lambda_val <= 1.0:
        return 32
    if lambda_val <= 4.0:
        return 64
    if lambda_val <= 10.0:
        return 96
    return 160


@lru_cache(maxsize=64)
def poisson_output_for_lambda(lambda_val: float) -> StatefulOutput:
    max_k = small_poisson_max_k(lambda_val) if lambda_val < 30.0 else 0
    return poisson_output(lambda_val, max_k)


@triton.jit
def transform_normal_u32(
    raw,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pairs = tl.reshape(raw, (BLOCK_SIZE // 2, 2))
    raw0, raw1 = tl.split(pairs)
    normal0, normal1 = uniform_to_normal_fast_f32(
        uint32_to_uniform(raw0),
        uint32_to_uniform(raw1),
    )

    value0 = mean + stddev * normal0
    value1 = mean + stddev * normal1
    if LOGNORMAL:
        value0 = tl.exp(value0)
        value1 = tl.exp(value1)

    return tl.reshape(tl.join(value0, value1), (BLOCK_SIZE,))


@triton.jit
def transform_poisson_small_u32(raw, lambda_val, MAX_K: tl.constexpr):
    return _poisson_inverse_from_uniform(uint32_to_uniform(raw), lambda_val, MAX_K)


@triton.jit
def transform_poisson_table_u32(
    raw,
    cdf_ptr,
    table_size,
    STEPS: tl.constexpr,
):
    return _poisson_table_lookup(uint32_to_uniform(raw), cdf_ptr, table_size, STEPS)


@triton.jit
def transform_poisson_large_u32(raw, lambda_val, BLOCK_SIZE: tl.constexpr):
    pairs = tl.reshape(raw, (BLOCK_SIZE // 2, 2))
    raw0, raw1 = tl.split(pairs)
    normal0, normal1 = uniform_to_normal_fast_f32(
        uint32_to_uniform(raw0),
        uint32_to_uniform(raw1),
    )
    sigma = tl.sqrt(lambda_val)
    value0 = tl.maximum(0, tl.floor(lambda_val + sigma * normal0 + 0.5)).to(tl.int32)
    value1 = tl.maximum(0, tl.floor(lambda_val + sigma * normal1 + 0.5)).to(tl.int32)

    return tl.reshape(tl.join(value0, value1), (BLOCK_SIZE,))


@triton.jit
def _poisson_inverse_from_uniform(u, lambda_val, MAX_K: tl.constexpr):
    probability = tl.exp(-lambda_val)
    cumulative = probability
    result = tl.full(u.shape, 0, tl.int32)
    for value in range(1, MAX_K + 1):
        active = u > cumulative
        probability = probability * lambda_val / value
        cumulative += probability
        result = tl.where(active, value, result)
    return result
