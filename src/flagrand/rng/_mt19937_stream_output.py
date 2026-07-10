from __future__ import annotations

import triton
import triton.language as tl

from flagrand.fused._internal.philox_poisson_table import _poisson_table_lookup
from flagrand.fused._internal.transforms import uint32_to_uniform
from flagrand.rng._stateful_output import transform_normal_u32, transform_poisson_large_u32


@triton.jit
def store_mt19937_outputs(
    out_ptr,
    prefix_raw,
    generated_raw,
    prefix_offsets,
    generated_offsets,
    prefix_mask,
    generated_mask,
    mean,
    stddev,
    lambda_val,
    poisson_cdf_ptr,
    poisson_table_size,
    OUTPUT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_K: tl.constexpr,
    POISSON_STEPS: tl.constexpr,
):
    if OUTPUT_MODE == 0:
        tl.store(out_ptr + prefix_offsets, prefix_raw.to(tl.int32, bitcast=True), mask=prefix_mask)
        tl.store(
            out_ptr + generated_offsets,
            generated_raw.to(tl.int32, bitcast=True),
            mask=generated_mask,
        )
    elif OUTPUT_MODE == 1:
        prefix_value = tl.maximum(
            tl.uint_to_uniform_float(prefix_raw),
            2.3283064365386963e-10,
        )
        generated_value = tl.maximum(
            tl.uint_to_uniform_float(generated_raw),
            2.3283064365386963e-10,
        )
        tl.store(out_ptr + prefix_offsets, prefix_value, mask=prefix_mask)
        tl.store(out_ptr + generated_offsets, generated_value, mask=generated_mask)
    elif OUTPUT_MODE == 2 or OUTPUT_MODE == 3:
        prefix_value = transform_normal_u32(
            prefix_raw, mean, stddev, OUTPUT_MODE == 3, BLOCK_SIZE
        )
        generated_value = transform_normal_u32(
            generated_raw, mean, stddev, OUTPUT_MODE == 3, BLOCK_SIZE
        )
        tl.store(out_ptr + prefix_offsets, prefix_value, mask=prefix_mask)
        tl.store(out_ptr + generated_offsets, generated_value, mask=generated_mask)
    elif OUTPUT_MODE == 4:
        prefix_value = _poisson_table_lookup(
            uint32_to_uniform(prefix_raw),
            poisson_cdf_ptr,
            poisson_table_size,
            POISSON_STEPS,
        )
        generated_value = _poisson_table_lookup(
            uint32_to_uniform(generated_raw),
            poisson_cdf_ptr,
            poisson_table_size,
            POISSON_STEPS,
        )
        tl.store(out_ptr + prefix_offsets, prefix_value, mask=prefix_mask)
        tl.store(out_ptr + generated_offsets, generated_value, mask=generated_mask)
    else:
        prefix_value = transform_poisson_large_u32(prefix_raw, lambda_val, BLOCK_SIZE)
        generated_value = transform_poisson_large_u32(generated_raw, lambda_val, BLOCK_SIZE)
        tl.store(out_ptr + prefix_offsets, prefix_value, mask=prefix_mask)
        tl.store(out_ptr + generated_offsets, generated_value, mask=generated_mask)
