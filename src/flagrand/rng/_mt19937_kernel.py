from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.rng._mt19937_data import MT19937_M, MT19937_N
from flagrand.rng._stateful_output import (
    StatefulOutput,
    transform_normal_u32,
    transform_poisson_large_u32,
    transform_poisson_small_u32,
)
from flagrand.runtime import CachedKernelLauncher


@triton.jit
def _mt19937_temper(y):
    full_mask = ~tl.zeros_like(y)
    y = y ^ ((y >> 11) & full_mask)
    y = y ^ ((y << 7) & 0x9D2C5680)
    y = y ^ ((y << 15) & 0xEFC60000)
    y = y ^ (y >> 18)
    return y


@triton.jit
def _mt19937_uint32_to_uniform(x):
    u = tl.uint_to_uniform_float(x.to(tl.uint32, bitcast=True))
    return tl.maximum(u, 2.3283064365386963e-10)


@triton.jit
def _mt19937_blocks_kernel(
    out_ptr,
    state_ptr,
    scratch_ptr,
    n_elements,
    n_blocks,
    mean,
    stddev,
    lambda_val,
    NUM_ROUNDS: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    M: tl.constexpr,
    MAX_K: tl.constexpr,
):
    stream_id = tl.program_id(0)
    state_base = stream_id * N
    scratch_base = stream_id * N

    tid = tl.arange(0, BLOCK_STATE)
    state_mask = tid < N
    state = tl.load(state_ptr + state_base + tid, mask=state_mask, other=0).to(
        tl.uint32, bitcast=True
    )

    upper_mask = tl.full((), 0x80000000, tl.uint32)
    lower_mask = tl.full((), 0x7FFFFFFF, tl.uint32)
    matrix_a = tl.full((), 0x9908B0DF, tl.uint32)
    one_u32 = tl.full((), 1, tl.uint32)
    zero_u32 = tl.full((), 0, tl.uint32)

    next_idx = tl.where(tid + 1 == N, 0, tid + 1)
    m_idx = tl.where(tid + M >= N, tid + M - N, tid + M)

    for round_idx in range(NUM_ROUNDS):
        tl.store(scratch_ptr + scratch_base + tid, state.to(tl.int32, bitcast=True), mask=state_mask)
        tl.debug_barrier()
        s_next = tl.load(scratch_ptr + scratch_base + next_idx, mask=state_mask, other=0).to(
            tl.uint32, bitcast=True
        )
        s_m = tl.load(scratch_ptr + scratch_base + m_idx, mask=state_mask, other=0).to(
            tl.uint32, bitcast=True
        )

        y = (state & upper_mask) | (s_next & lower_mask)
        mag = tl.where((y & one_u32) != zero_u32, matrix_a, zero_u32)
        state = s_m ^ (y >> 1) ^ mag

        tempered = _mt19937_temper(state)
        out_offsets = (round_idx * n_blocks + stream_id) * N + tid
        out_mask = state_mask & (out_offsets < n_elements)
        if OUTPUT_MODE == 1:
            tl.store(out_ptr + out_offsets, _mt19937_uint32_to_uniform(tempered), mask=out_mask)
        elif OUTPUT_MODE == 0:
            tl.store(out_ptr + out_offsets, tempered.to(tl.int32, bitcast=True), mask=out_mask)
        elif OUTPUT_MODE == 2 or OUTPUT_MODE == 3:
            transformed = transform_normal_u32(
                tempered, mean, stddev, OUTPUT_MODE == 3, BLOCK_STATE
            )
            tl.store(out_ptr + out_offsets, transformed, mask=out_mask)
        elif OUTPUT_MODE == 4:
            transformed = transform_poisson_small_u32(tempered, lambda_val, MAX_K)
            tl.store(out_ptr + out_offsets, transformed, mask=out_mask)
        else:
            transformed = transform_poisson_large_u32(tempered, lambda_val, BLOCK_STATE)
            tl.store(out_ptr + out_offsets, transformed, mask=out_mask)

        if round_idx + 1 < NUM_ROUNDS:
            tl.debug_barrier()

    tl.store(state_ptr + state_base + tid, state.to(tl.int32, bitcast=True), mask=state_mask)


@triton.jit
def _mt19937_single_round_kernel(
    out_ptr,
    state_ptr,
    mean,
    stddev,
    lambda_val,
    OUTPUT_MODE: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
    N: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    M: tl.constexpr,
    MAX_K: tl.constexpr,
):
    stream_id = tl.program_id(0)
    state_base = stream_id * N

    tid = tl.arange(0, BLOCK_STATE)
    state_mask = tid < N

    upper_mask = tl.full((), 0x80000000, tl.uint32)
    lower_mask = tl.full((), 0x7FFFFFFF, tl.uint32)
    matrix_a = tl.full((), 0x9908B0DF, tl.uint32)
    one_u32 = tl.full((), 1, tl.uint32)
    zero_u32 = tl.full((), 0, tl.uint32)

    next_idx = tl.where(tid + 1 == N, 0, tid + 1)
    m_idx = tl.where(tid + M >= N, tid + M - N, tid + M)

    state = tl.load(state_ptr + state_base + tid, mask=state_mask, other=0).to(
        tl.uint32, bitcast=True
    )
    s_next = tl.load(state_ptr + state_base + next_idx, mask=state_mask, other=0).to(
        tl.uint32, bitcast=True
    )
    s_m = tl.load(state_ptr + state_base + m_idx, mask=state_mask, other=0).to(
        tl.uint32, bitcast=True
    )
    tl.debug_barrier()

    y = (state & upper_mask) | (s_next & lower_mask)
    mag = tl.where((y & one_u32) != zero_u32, matrix_a, zero_u32)
    new_state = s_m ^ (y >> 1) ^ mag
    tempered = _mt19937_temper(new_state)

    if WRITE_OUTPUT:
        out_offsets = stream_id * N + tid
        if OUTPUT_MODE == 1:
            tl.store(out_ptr + out_offsets, _mt19937_uint32_to_uniform(tempered), mask=state_mask)
        elif OUTPUT_MODE == 0:
            tl.store(out_ptr + out_offsets, tempered.to(tl.int32, bitcast=True), mask=state_mask)
        elif OUTPUT_MODE == 2 or OUTPUT_MODE == 3:
            transformed = transform_normal_u32(
                tempered, mean, stddev, OUTPUT_MODE == 3, BLOCK_STATE
            )
            tl.store(out_ptr + out_offsets, transformed, mask=state_mask)
        elif OUTPUT_MODE == 4:
            transformed = transform_poisson_small_u32(tempered, lambda_val, MAX_K)
            tl.store(out_ptr + out_offsets, transformed, mask=state_mask)
        else:
            transformed = transform_poisson_large_u32(tempered, lambda_val, BLOCK_STATE)
            tl.store(out_ptr + out_offsets, transformed, mask=state_mask)
    tl.store(state_ptr + state_base + tid, new_state.to(tl.int32, bitcast=True), mask=state_mask)


_MT19937_BLOCKS_LAUNCHER = CachedKernelLauncher(
    _mt19937_blocks_kernel,
    constexpr_names=("NUM_ROUNDS", "OUTPUT_MODE", "N", "BLOCK_STATE", "M", "MAX_K"),
)
_MT19937_SINGLE_ROUND_LAUNCHER = CachedKernelLauncher(
    _mt19937_single_round_kernel,
    constexpr_names=("OUTPUT_MODE", "WRITE_OUTPUT", "N", "BLOCK_STATE", "M", "MAX_K"),
)


def launch_mt19937_blocks(
    out: torch.Tensor,
    state: torch.Tensor,
    scratch: torch.Tensor,
    *,
    output_elements: int,
    rounds: int,
    block_count: int,
    output: StatefulOutput,
    requested_num_warps: int,
) -> None:
    grid = (block_count,)
    if rounds == 1:
        _MT19937_SINGLE_ROUND_LAUNCHER.launch(
            grid,
            (out, state, output.mean, output.stddev, output.lambda_val),
            (output.mode, output_elements > 0, MT19937_N, 1024, MT19937_M, output.max_k),
            (mt19937_launch_warps(block_count, requested_num_warps),),
        )
        return

    _MT19937_BLOCKS_LAUNCHER.launch(
        grid,
        (
            out,
            state,
            scratch,
            int(output_elements),
            block_count,
            output.mean,
            output.stddev,
            output.lambda_val,
        ),
        (int(rounds), output.mode, MT19937_N, 1024, MT19937_M, output.max_k),
        (mt19937_launch_warps(block_count, requested_num_warps),),
    )


def mt19937_launch_warps(block_count: int, requested: int) -> int:
    if block_count <= 32:
        return min(requested, 2)
    return requested
