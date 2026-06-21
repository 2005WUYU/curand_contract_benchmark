from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from flagrand.rng._sequence import clear_chunk_cache, generate_chunked

_params = torch.load(str(Path(__file__).parent / "data" / "mtgp32_params.pt"), map_location="cpu")

_MTGP32_BLOCK_SIZE = 256
_MTGP32_MAX_BLOCKS = 192
_SEQUENCE_CHUNK = _MTGP32_BLOCK_SIZE * _MTGP32_MAX_BLOCKS
_MTGPDC_N = 351
_MTGP32_STATE_SIZE = 1024
_MTGP32_STATE_MASK = 1023
_MTGP32_MASK = 0xFFF80000


def _u32_to_int32(buf: torch.Tensor) -> torch.Tensor:
    return torch.where(buf >= 0x80000000, buf - 0x100000000, buf).to(torch.int32)


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
def _mtgp32_kernel(
    out_ptr,
    state_ptr,
    pos_ptr,
    sh1_ptr,
    sh2_ptr,
    param_ptr,
    temper_ptr,
    n_elements,
    num_iters,
    n_blocks,
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

    for k in range(num_iters):
        STATE_OFFSET = (k * BLOCK_SIZE) & STATE_MASK

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

        out_idx = (k * n_blocks + pid) * BLOCK_SIZE + offs
        out_mask = out_idx < n_elements
        tl.store(out_ptr + out_idx, o.to(tl.int32, bitcast=True), mask=out_mask)

        tl.debug_barrier()


def _mtgp32_init_state_cpu(bid: int, state_seed: int) -> list[int]:
    hidden_seed = int(_params["hidden_seeds"][bid % 200].item())

    tmp = hidden_seed
    tmp = (tmp + (tmp >> 16)) & 0xFFFFFFFF
    tmp = (tmp + (tmp >> 8)) & 0xFFFFFFFF
    fill_val = (tmp & 0xFF) * 0x01010101

    state = [fill_val] * _MTGPDC_N
    state[0] = state_seed & 0xFFFFFFFF
    state[1] = hidden_seed

    for i in range(1, _MTGPDC_N):
        prev = state[i - 1]
        state[i] = (state[i] ^ ((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)) & 0xFFFFFFFF

    return state


@lru_cache(maxsize=32)
def _build_initial_state(seed: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    s = (seed ^ (seed >> 32)) & 0xFFFFFFFF

    buf = torch.zeros((_MTGP32_MAX_BLOCKS, _MTGP32_STATE_SIZE), dtype=torch.int64)
    for bid in range(_MTGP32_MAX_BLOCKS):
        state_seed = (s + bid + 1) & 0xFFFFFFFF
        st = _mtgp32_init_state_cpu(bid, state_seed)
        buf[bid, :_MTGPDC_N] = torch.tensor(st, dtype=torch.int64)

    return _u32_to_int32(buf).to(device=device).contiguous()


@lru_cache(maxsize=4)
def _build_param_tensors(device_str: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)
    pos = _params["pos"][:_MTGP32_MAX_BLOCKS].to(device=device)
    sh1 = _params["sh1"][:_MTGP32_MAX_BLOCKS].to(device=device)
    sh2 = _params["sh2"][:_MTGP32_MAX_BLOCKS].to(device=device)
    param = _u32_to_int32(_params["param"][: _MTGP32_MAX_BLOCKS * 16]).to(device=device)
    temper = _u32_to_int32(_params["temper"][: _MTGP32_MAX_BLOCKS * 16]).to(device=device)
    return pos, sh1, sh2, param, temper


@dataclass
class Mtgp32Generator:
    seed: int = 0
    offset: int = 0

    @property
    def dimensions(self) -> None:
        return None

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        offset_val = self.offset if offset is None else int(offset)
        if offset is not None and offset_val != self.offset:
            raise ValueError("MTGP32 explicit offset override is not supported.")
        seed_val = self.seed if seed is None else int(seed)
        block_size = int(kwargs.get("block_size", _MTGP32_BLOCK_SIZE))
        if block_size != _MTGP32_BLOCK_SIZE:
            raise ValueError("MTGP32 uses a fixed block_size=256 to preserve per-state dependency ordering.")
        num_warps = kwargs.get("num_warps", 8)

        n = out.numel()
        if n == 0:
            return out
        if seed is not None:
            raise ValueError("MTGP32 explicit seed override is not supported.")

        generate_chunked(
            self,
            out,
            start=offset_val,
            chunk_size=_SEQUENCE_CHUNK,
            cache_key=(seed_val, str(out.device), str(out.dtype), num_warps),
            generate_chunk=lambda chunk, chunk_start: self._generate_chunk(chunk, seed_val, num_warps),
        )
        self.offset = offset_val + n
        return out

    def _generate_chunk(self, out: torch.Tensor, seed_val: int, num_warps: int) -> None:
        device_str = str(out.device)
        pos, sh1, sh2, param, temper = _build_param_tensors(device_str)

        blocks_needed = (out.numel() + _MTGP32_BLOCK_SIZE - 1) // _MTGP32_BLOCK_SIZE
        n_blocks = min(_MTGP32_MAX_BLOCKS, blocks_needed)
        num_iters = (blocks_needed + n_blocks - 1) // n_blocks

        # Mutated in place so subsequent calls advance the stream.
        ws_seed = getattr(self, "_ws_seed", None)
        ws_device = getattr(self, "_ws_device", None)
        ws_blocks = getattr(self, "_ws_blocks", 0)
        if ws_seed != seed_val or ws_device != device_str or ws_blocks < n_blocks:
            initial_state = _build_initial_state(seed_val, device_str)
            self._working_state = initial_state.clone()
            self._ws_seed = seed_val
            self._ws_device = device_str
            self._ws_blocks = self._working_state.shape[0]
            clear_chunk_cache(self)
        state = self._working_state[:n_blocks]

        grid = (n_blocks,)
        _mtgp32_kernel[grid](
            out,
            state,
            pos[:n_blocks],
            sh1[:n_blocks],
            sh2[:n_blocks],
            param[: n_blocks * 16],
            temper[: n_blocks * 16],
            out.numel(),
            num_iters,
            n_blocks,
            BLOCK_SIZE=_MTGP32_BLOCK_SIZE,
            STATE_MASK=_MTGP32_STATE_MASK,
            MASK=_MTGP32_MASK,
            N_RECUR=_MTGPDC_N,
            num_warps=num_warps,
        )
