from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

PARAMS = torch.load(str(Path(__file__).parent / "data" / "mtgp32_params.pt"), map_location="cpu")

MTGP32_BLOCK_SIZE = 256
MTGP32_MAX_BLOCKS = 192
MTGP32_SEQUENCE_CHUNK = MTGP32_BLOCK_SIZE * MTGP32_MAX_BLOCKS
MTGPDC_N = 351
MTGP32_STATE_SIZE = 1024
MTGP32_STATE_MASK = 1023
MTGP32_MASK = 0xFFF80000
MTGP32_MAX_CHUNKS_PER_LAUNCH = 256


def u32_to_int32(buf: torch.Tensor) -> torch.Tensor:
    return torch.where(buf >= 0x80000000, buf - 0x100000000, buf).to(torch.int32)


def mtgp32_init_state_cpu(bid: int, state_seed: int) -> list[int]:
    hidden_seed = int(PARAMS["hidden_seeds"][bid % 200].item())

    tmp = hidden_seed
    tmp = (tmp + (tmp >> 16)) & 0xFFFFFFFF
    tmp = (tmp + (tmp >> 8)) & 0xFFFFFFFF
    fill_val = (tmp & 0xFF) * 0x01010101

    state = [fill_val] * MTGPDC_N
    state[0] = state_seed & 0xFFFFFFFF
    state[1] = hidden_seed

    for i in range(1, MTGPDC_N):
        prev = state[i - 1]
        state[i] = (state[i] ^ ((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)) & 0xFFFFFFFF

    return state


@lru_cache(maxsize=32)
def build_initial_state(seed: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    s = (seed ^ (seed >> 32)) & 0xFFFFFFFF

    buf = torch.zeros((MTGP32_MAX_BLOCKS, MTGP32_STATE_SIZE), dtype=torch.int64)
    for bid in range(MTGP32_MAX_BLOCKS):
        state_seed = (s + bid + 1) & 0xFFFFFFFF
        st = mtgp32_init_state_cpu(bid, state_seed)
        buf[bid, :MTGPDC_N] = torch.tensor(st, dtype=torch.int64)

    return u32_to_int32(buf).to(device=device).contiguous()


@lru_cache(maxsize=4)
def build_param_tensors(device_str: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)
    pos = PARAMS["pos"][:MTGP32_MAX_BLOCKS].to(device=device)
    sh1 = PARAMS["sh1"][:MTGP32_MAX_BLOCKS].to(device=device)
    sh2 = PARAMS["sh2"][:MTGP32_MAX_BLOCKS].to(device=device)
    param = u32_to_int32(PARAMS["param"][: MTGP32_MAX_BLOCKS * 16]).to(device=device)
    temper = u32_to_int32(PARAMS["temper"][: MTGP32_MAX_BLOCKS * 16]).to(device=device)
    return pos, sh1, sh2, param, temper
