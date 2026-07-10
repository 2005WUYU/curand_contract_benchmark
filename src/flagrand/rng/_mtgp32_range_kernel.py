from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.philox_poisson_table import _cdf_table
from flagrand.rng._mtgp32_data import (
    MTGP32_BLOCK_SIZE,
    MTGP32_MASK,
    MTGP32_MAX_BLOCKS,
    MTGP32_STATE_MASK,
    MTGPDC_N,
)
from flagrand.rng._mtgp32_kernel import (
    _mtgp32_recurrence,
    _mtgp32_temper,
    _mtgp32_uint32_to_uniform,
    mtgp32_launch_warps,
)
from flagrand.rng._stateful_output import (
    StatefulOutput,
    transform_normal_u32,
    transform_poisson_large_u32,
    transform_poisson_table_u32,
)
from flagrand.runtime import CachedKernelLauncher


@triton.jit
def _mtgp32_range_kernel(
    out_ptr,
    state_ptr,
    pos_ptr,
    sh1_ptr,
    sh2_ptr,
    param_ptr,
    n_elements,
    start_stream,
    base_iter,
    mean,
    stddev,
    lambda_val,
    poisson_cdf_ptr,
    poisson_table_size,
    OUTPUT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    MAX_K: tl.constexpr,
    POISSON_STEPS: tl.constexpr,
):
    output_block = tl.program_id(0)
    unwrapped_stream = start_stream + output_block
    wrapped = unwrapped_stream >= MAX_BLOCKS
    stream_id = tl.where(wrapped, unwrapped_stream - MAX_BLOCKS, unwrapped_stream)
    state_iter = base_iter + wrapped

    pos = tl.load(pos_ptr + stream_id)
    sh1 = tl.load(sh1_ptr + stream_id)
    sh2 = tl.load(sh2_ptr + stream_id)
    state_base = stream_id * (STATE_MASK + 1)
    param_base = stream_id * 16
    offsets = tl.arange(0, BLOCK_SIZE)
    state_offset = (state_iter * BLOCK_SIZE) & STATE_MASK

    x1 = tl.load(state_ptr + state_base + ((offsets + state_offset) & STATE_MASK)).to(
        tl.uint32, bitcast=True
    )
    x2 = tl.load(state_ptr + state_base + ((offsets + state_offset + 1) & STATE_MASK)).to(
        tl.uint32, bitcast=True
    )
    y = tl.load(state_ptr + state_base + ((offsets + state_offset + pos) & STATE_MASK)).to(
        tl.uint32, bitcast=True
    )
    y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
    matrix = tl.load(param_ptr + param_base + (y & 0x0F)).to(tl.uint32, bitcast=True)
    state_value = y ^ matrix
    new_state_index = (offsets + state_offset + N_RECUR) & STATE_MASK
    tl.store(
        state_ptr + state_base + new_state_index,
        state_value.to(tl.int32, bitcast=True),
    )

    temper_state = tl.load(
        state_ptr + state_base + ((offsets + state_offset + pos - 1) & STATE_MASK)
    ).to(tl.uint32, bitcast=True)
    raw = _mtgp32_temper(state_value, temper_state)
    output_offsets = output_block * BLOCK_SIZE + offsets
    output_mask = output_offsets < n_elements
    if OUTPUT_MODE == 1:
        tl.store(out_ptr + output_offsets, _mtgp32_uint32_to_uniform(raw), mask=output_mask)
    elif OUTPUT_MODE == 0:
        tl.store(out_ptr + output_offsets, raw.to(tl.int32, bitcast=True), mask=output_mask)
    elif OUTPUT_MODE == 2 or OUTPUT_MODE == 3:
        transformed = transform_normal_u32(
            raw, mean, stddev, OUTPUT_MODE == 3, BLOCK_SIZE
        )
        tl.store(out_ptr + output_offsets, transformed, mask=output_mask)
    elif OUTPUT_MODE == 4:
        transformed = transform_poisson_table_u32(
            raw, poisson_cdf_ptr, poisson_table_size, POISSON_STEPS
        )
        tl.store(out_ptr + output_offsets, transformed, mask=output_mask)
    else:
        transformed = transform_poisson_large_u32(raw, lambda_val, BLOCK_SIZE)
        tl.store(out_ptr + output_offsets, transformed, mask=output_mask)


_MTGP32_RANGE_LAUNCHER = CachedKernelLauncher(
    _mtgp32_range_kernel,
    constexpr_names=(
        "OUTPUT_MODE",
        "BLOCK_SIZE",
        "MAX_BLOCKS",
        "STATE_MASK",
        "MASK",
        "N_RECUR",
        "MAX_K",
        "POISSON_STEPS",
    ),
)


def generate_range_mtgp32(
    generator,
    out: torch.Tensor,
    *,
    block_offset: int,
    num_warps: int,
    output: StatefulOutput,
) -> None:
    num_blocks = out.numel() // MTGP32_BLOCK_SIZE
    start_stream = block_offset % MTGP32_MAX_BLOCKS
    base_iter = (block_offset // MTGP32_MAX_BLOCKS) % 4
    pos, sh1, sh2, param, _ = generator._parameter_tensors
    if output.mode == 4:
        poisson_cache = getattr(generator, "_mtgp32_poisson_cache", None)
        if poisson_cache is None or poisson_cache[0] != output.lambda_val:
            poisson_table = _cdf_table(output.lambda_val, str(out.device))
            poisson_cache = (
                output.lambda_val,
                poisson_table,
                math.ceil(math.log2(poisson_table.numel())),
            )
            generator._mtgp32_poisson_cache = poisson_cache
        _, poisson_table, poisson_steps = poisson_cache
    else:
        poisson_table = generator._working_state
        poisson_steps = 0
    launch_warps = mtgp32_launch_warps(num_blocks, 1, num_warps)
    runtime_ints = (out.numel(), start_stream, base_iter, poisson_table.numel())
    _MTGP32_RANGE_LAUNCHER.launch(
        (num_blocks,),
        (
            out,
            generator._working_state,
            pos,
            sh1,
            sh2,
            param,
            runtime_ints[0],
            runtime_ints[1],
            runtime_ints[2],
            output.mean,
            output.stddev,
            output.lambda_val,
            poisson_table,
            runtime_ints[3],
        ),
        (
            output.mode,
            MTGP32_BLOCK_SIZE,
            MTGP32_MAX_BLOCKS,
            MTGP32_STATE_MASK,
            MTGP32_MASK,
            MTGPDC_N,
            output.max_k,
            poisson_steps,
        ),
        (launch_warps,),
        specialization_key=(
            str(out.device),
            str(out.dtype),
            output.mode,
            output.max_k,
            poisson_steps,
            tuple((value == 1, value % 16 == 0) for value in runtime_ints),
            num_blocks,
            launch_warps,
        ),
    )
    generator._ws_next_block_start = (block_offset + num_blocks) * MTGP32_BLOCK_SIZE
