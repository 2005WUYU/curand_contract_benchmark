from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.philox_poisson_table import _cdf_table
from flagrand.rng._mt19937_data import MT19937_M, MT19937_N, MT19937_NUM_STREAMS
from flagrand.rng._mt19937_kernel import _mt19937_temper, mt19937_launch_warps
from flagrand.rng._mt19937_stream_output import store_mt19937_outputs
from flagrand.rng._stateful_output import StatefulOutput
from flagrand.runtime import CachedKernelLauncher


@triton.jit
def _mt19937_stream_kernel(
    out_ptr,
    state_ptr,
    prefix_raw_ptr,
    next_raw_ptr,
    n_elements,
    prefix_offset,
    prefix_count,
    block_count,
    start_stream,
    mean,
    stddev,
    lambda_val,
    poisson_cdf_ptr,
    poisson_table_size,
    OUTPUT_MODE: tl.constexpr,
    N: tl.constexpr,
    NUM_STREAMS: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    M: tl.constexpr,
    MAX_K: tl.constexpr,
    POISSON_STEPS: tl.constexpr,
):
    program_id = tl.program_id(0)
    tid = tl.arange(0, BLOCK_STATE)
    state_mask = tid < N
    stream_active = program_id < block_count

    stream_id = start_stream + program_id
    stream_id = tl.where(stream_id >= NUM_STREAMS, stream_id - NUM_STREAMS, stream_id)
    state_base = stream_id * N

    upper_mask = tl.full((), 0x80000000, tl.uint32)
    lower_mask = tl.full((), 0x7FFFFFFF, tl.uint32)
    matrix_a = tl.full((), 0x9908B0DF, tl.uint32)
    one_u32 = tl.full((), 1, tl.uint32)
    zero_u32 = tl.full((), 0, tl.uint32)
    next_index = tl.where(tid + 1 == N, 0, tid + 1)
    m_index = tl.where(tid + M >= N, tid + M - N, tid + M)
    state = tl.load(
        state_ptr + state_base + tid,
        mask=state_mask & stream_active,
        other=0,
    ).to(tl.uint32, bitcast=True)
    state_next = tl.load(
        state_ptr + state_base + next_index,
        mask=state_mask & stream_active,
        other=0,
    ).to(tl.uint32, bitcast=True)
    state_m = tl.load(
        state_ptr + state_base + m_index,
        mask=state_mask & stream_active,
        other=0,
    ).to(tl.uint32, bitcast=True)
    y = (state & upper_mask) | (state_next & lower_mask)
    magnitude = tl.where((y & one_u32) != zero_u32, matrix_a, zero_u32)
    new_state = state_m ^ (y >> 1) ^ magnitude
    generated_raw = _mt19937_temper(new_state)
    tl.store(
        state_ptr + state_base + tid,
        new_state.to(tl.int32, bitcast=True),
        mask=state_mask & stream_active,
    )

    prefix_raw = tl.load(
        prefix_raw_ptr + prefix_offset + tid,
        mask=(program_id == 0) & (tid < prefix_count),
        other=0,
    ).to(tl.uint32, bitcast=True)
    prefix_mask = (program_id == 0) & (tid < prefix_count) & (tid < n_elements)
    generated_offsets = prefix_count + program_id * N + tid
    generated_mask = state_mask & stream_active & (generated_offsets < n_elements)
    store_mt19937_outputs(
        out_ptr,
        prefix_raw,
        generated_raw,
        tid,
        generated_offsets,
        prefix_mask,
        generated_mask,
        mean,
        stddev,
        lambda_val,
        poisson_cdf_ptr,
        poisson_table_size,
        OUTPUT_MODE,
        BLOCK_STATE,
        MAX_K,
        POISSON_STEPS,
    )

    is_last_stream = stream_active & (program_id + 1 == block_count)
    tl.store(
        next_raw_ptr + tid,
        generated_raw.to(tl.int32, bitcast=True),
        mask=state_mask & is_last_stream,
    )


_MT19937_STREAM_LAUNCHER = CachedKernelLauncher(
    _mt19937_stream_kernel,
    constexpr_names=(
        "OUTPUT_MODE",
        "N",
        "NUM_STREAMS",
        "BLOCK_STATE",
        "M",
        "MAX_K",
        "POISSON_STEPS",
    ),
)


def launch_mt19937_stream(
    generator,
    out: torch.Tensor,
    *,
    prefix_offset: int,
    prefix_count: int,
    block_count: int,
    start_stream: int,
    output: StatefulOutput,
    num_warps: int,
) -> None:
    grid = (max(1, block_count),)
    n_elements = out.numel()
    runtime_ints = (n_elements, prefix_offset, prefix_count, block_count, start_stream)
    if output.mode == 4:
        poisson_cache = getattr(generator, "_mt19937_poisson_cache", None)
        if poisson_cache is None or poisson_cache[0] != output.lambda_val:
            poisson_table = _cdf_table(output.lambda_val, str(out.device))
            poisson_cache = (
                output.lambda_val,
                poisson_table,
                math.ceil(math.log2(poisson_table.numel())),
            )
            generator._mt19937_poisson_cache = poisson_cache
        _, poisson_table, poisson_steps = poisson_cache
    else:
        poisson_table = generator._mt19937_prefix_raw
        poisson_steps = 0
    launch_warps = mt19937_launch_warps(block_count, num_warps)
    _MT19937_STREAM_LAUNCHER.launch(
        grid,
        (
            out,
            generator._working_state,
            generator._mt19937_prefix_raw,
            generator._mt19937_next_raw,
            *runtime_ints,
            output.mean,
            output.stddev,
            output.lambda_val,
            poisson_table,
            poisson_table.numel(),
        ),
        (
            output.mode,
            MT19937_N,
            MT19937_NUM_STREAMS,
            1024,
            MT19937_M,
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
            (poisson_table.numel() == 1, poisson_table.numel() % 16 == 0),
            (
                (n_elements == 1, n_elements % 16 == 0),
                (prefix_offset == 1, prefix_offset % 16 == 0),
                (prefix_count == 1, prefix_count % 16 == 0),
                (block_count == 1, block_count % 16 == 0),
                (start_stream == 1, start_stream % 16 == 0),
            ),
            grid,
            launch_warps,
        ),
    )
