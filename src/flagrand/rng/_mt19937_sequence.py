from __future__ import annotations

import torch

from flagrand.rng._mt19937_state import ensure_working_state
from flagrand.rng._mt19937_stream import generate_streamed_mt19937
from flagrand.rng._stateful_output import RAW_OUTPUT, StatefulOutput


def generate_mt19937_contiguous(
    generator,
    out: torch.Tensor,
    seed_val: int,
    offset_val: int,
    num_warps: int,
    *,
    output: StatefulOutput = RAW_OUTPUT,
) -> None:
    device_str = str(out.device)
    ensure_working_state(generator, seed_val, device_str)
    generate_streamed_mt19937(generator, out, offset_val, num_warps, output)
