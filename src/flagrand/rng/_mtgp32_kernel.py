from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.rng._mtgp32_data import (
    MTGP32_BLOCK_SIZE,
    MTGP32_MASK,
    MTGP32_STATE_MASK,
    MTGPDC_N,
)


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
    NUM_ITERS: tl.constexpr,
    START_ITER: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
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
        else:
            tl.store(out_ptr + out_idx, o.to(tl.int32, bitcast=True), mask=out_mask)

        if k + 1 < NUM_ITERS:
            tl.debug_barrier()


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
    output_mode: int,
    requested_num_warps: int,
) -> None:
    grid = (block_count,)
    _mtgp32_kernel[grid](
        out,
        state,
        pos,
        sh1,
        sh2,
        param,
        temper,
        output_elements,
        int(chunks),
        start_iter,
        block_count,
        output_mode,
        BLOCK_SIZE=MTGP32_BLOCK_SIZE,
        STATE_MASK=MTGP32_STATE_MASK,
        MASK=MTGP32_MASK,
        N_RECUR=MTGPDC_N,
        num_warps=mtgp32_launch_warps(block_count, chunks, requested_num_warps),
    )


def mtgp32_launch_warps(block_count: int, chunks: int, requested: int) -> int:
    if chunks == 1 and block_count <= 32:
        return min(requested, 2)
    return requested
