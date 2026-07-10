from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from flagrand.rng._mtgp32_data import (
    MTGP32_BLOCK_SIZE,
    MTGP32_MASK,
    MTGP32_STATE_MASK,
    MTGPDC_N,
)
from flagrand.rng._stateful_output import (
    StatefulOutput,
    transform_normal_u32,
    transform_poisson_large_u32,
    transform_poisson_table_u32,
)
from flagrand.fused._internal.philox_poisson_table import _cdf_table
from flagrand.runtime import CachedKernelLauncher


@triton.jit
def _mtgp32_recurrence(X1, X2, Y, sh1, sh2, MASK):
    X = (X1 & MASK) ^ X2
    X = X ^ ((X << sh1) & 0xFFFFFFFF)
    return X ^ (Y >> sh2)


@triton.jit
def _mtgp32_temper(r, T):
    T = T ^ (T >> 16)
    T = T ^ (T >> 8)
    return r ^ T


@triton.jit
def _mtgp32_uint32_to_uniform(x):
    u = tl.uint_to_uniform_float(x.to(tl.uint32, bitcast=True))
    return tl.maximum(u, 2.3283064365386963e-10)


@triton.jit
def _mtgp32_kernel(
    out_ptr,
    state_ptr,
    pos_ptr,
    sh1_ptr,
    sh2_ptr,
    param_ptr,
    temper_ptr,
    n_elements,
    mean,
    stddev,
    lambda_val,
    poisson_cdf_ptr,
    poisson_table_size,
    NUM_ITERS: tl.constexpr,
    START_ITER: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    MAX_K: tl.constexpr,
    POISSON_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(pos_ptr + pid)
    sh1 = tl.load(sh1_ptr + pid)
    sh2 = tl.load(sh2_ptr + pid)

    s_base = pid * (STATE_MASK + 1)
    p_base = pid * 16
    offs = tl.arange(0, BLOCK_SIZE)

    for k in range(NUM_ITERS):
        STATE_OFFSET = ((START_ITER + k) * BLOCK_SIZE) & STATE_MASK

        X1 = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET) & STATE_MASK)).to(tl.uint32, bitcast=True)
        X2 = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        Y = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)

        Y = _mtgp32_recurrence(X1, X2, Y, sh1, sh2, MASK)

        MAT = tl.load(param_ptr + p_base + (Y & 0x0F)).to(tl.uint32, bitcast=True)
        r = Y ^ MAT

        new_state_idx = (offs + STATE_OFFSET + N_RECUR) & STATE_MASK
        tl.store(state_ptr + s_base + new_state_idx, r.to(tl.int32, bitcast=True))

        T = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        o = _mtgp32_temper(r, T)

        out_idx = (k * N_BLOCKS + pid) * BLOCK_SIZE + offs
        out_mask = out_idx < n_elements
        if OUTPUT_MODE == 1:
            tl.store(out_ptr + out_idx, _mtgp32_uint32_to_uniform(o), mask=out_mask)
        elif OUTPUT_MODE == 0:
            tl.store(out_ptr + out_idx, o.to(tl.int32, bitcast=True), mask=out_mask)
        elif OUTPUT_MODE == 2 or OUTPUT_MODE == 3:
            transformed = transform_normal_u32(o, mean, stddev, OUTPUT_MODE == 3, BLOCK_SIZE)
            tl.store(out_ptr + out_idx, transformed, mask=out_mask)
        elif OUTPUT_MODE == 4:
            transformed = transform_poisson_table_u32(
                o, poisson_cdf_ptr, poisson_table_size, POISSON_STEPS
            )
            tl.store(out_ptr + out_idx, transformed, mask=out_mask)
        else:
            transformed = transform_poisson_large_u32(o, lambda_val, BLOCK_SIZE)
            tl.store(out_ptr + out_idx, transformed, mask=out_mask)

        if k + 1 < NUM_ITERS:
            tl.debug_barrier()


_MTGP32_LAUNCHER = CachedKernelLauncher(
    _mtgp32_kernel,
    constexpr_names=(
        "NUM_ITERS",
        "START_ITER",
        "N_BLOCKS",
        "OUTPUT_MODE",
        "BLOCK_SIZE",
        "STATE_MASK",
        "MASK",
        "N_RECUR",
        "MAX_K",
        "POISSON_STEPS",
    ),
)


def launch_mtgp32_blocks(
    out: torch.Tensor,
    state: torch.Tensor,
    pos: torch.Tensor,
    sh1: torch.Tensor,
    sh2: torch.Tensor,
    param: torch.Tensor,
    temper: torch.Tensor,
    *,
    output_elements: int,
    chunks: int,
    start_iter: int,
    block_count: int,
    output: StatefulOutput,
    requested_num_warps: int,
) -> None:
    grid = (block_count,)
    poisson_table = _cdf_table(output.lambda_val, str(out.device)) if output.mode == 4 else state
    poisson_steps = math.ceil(math.log2(poisson_table.numel())) if output.mode == 4 else 0
    _MTGP32_LAUNCHER.launch(
        grid,
        (
            out,
            state,
            pos,
            sh1,
            sh2,
            param,
            temper,
            output_elements,
            output.mean,
            output.stddev,
            output.lambda_val,
            poisson_table,
            poisson_table.numel(),
        ),
        (
            int(chunks),
            start_iter,
            block_count,
            output.mode,
            MTGP32_BLOCK_SIZE,
            MTGP32_STATE_MASK,
            MTGP32_MASK,
            MTGPDC_N,
            output.max_k,
            poisson_steps,
        ),
        (mtgp32_launch_warps(block_count, chunks, requested_num_warps),),
    )


def mtgp32_launch_warps(block_count: int, chunks: int, requested: int) -> int:
    if chunks == 1 and block_count <= 32:
        return min(requested, 2)
    return requested
