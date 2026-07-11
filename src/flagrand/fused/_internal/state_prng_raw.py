from __future__ import annotations

import torch

from flagrand.fused._internal.state_prng_kernels import raw_state_kernel
from flagrand.fused._internal.state_prng_state import RNG_MRG32K3A, RNG_XORWOW
from flagrand.runtime import CachedKernelLauncher

_BLOCK: int = 128
_STATE_THREADS: int = 16384
_STATE_ROWS: int = 6
_CACHE_NAMES = (
    "_raw_sequence_key",
    "_raw_sequence_state",
    "_raw_sequence_state_offset",
    "_raw_sequence_pending",
    "_raw_sequence_pending_offset",
)

_RAW_STATE_LAUNCHER = CachedKernelLauncher(
    raw_state_kernel,
    constexpr_names=("INIT_STATE", "STATE_THREADS", "BLOCK", "RNG_KIND"),
)


def generate_xorwow_raw(out: torch.Tensor, generator) -> None:
    _generate_raw(out, generator, RNG_XORWOW, "XORWOW")


def generate_mrg32k3a_raw(out: torch.Tensor, generator) -> None:
    _generate_raw(out, generator, RNG_MRG32K3A, "MRG32K3A")


def discard_state_prng_raw_sequence(generator) -> None:
    for name in _CACHE_NAMES:
        if hasattr(generator, name):
            delattr(generator, name)


def _generate_raw(out: torch.Tensor, generator, rng_kind: int, name: str) -> None:
    flat = out.view(-1)
    n = flat.numel()
    if n == 0:
        return

    offset = int(getattr(generator, "offset", 0))
    if offset < 0:
        raise ValueError(f"{name}: offset must be >= 0, got {offset}.")
    seed = int(getattr(generator, "seed", 0))
    key = (seed, str(out.device), str(out.dtype), rng_kind)
    if getattr(generator, "_raw_sequence_key", None) != key:
        discard_state_prng_raw_sequence(generator)
        generator._raw_sequence_key = key

    written = _consume_pending(flat, generator, offset)
    generator._raw_sequence_key = key
    offset += written
    remaining = n - written
    if remaining == 0:
        generator.offset = offset
        return

    state = getattr(generator, "_raw_sequence_state", None)
    state_offset = int(getattr(generator, "_raw_sequence_state_offset", -1))
    initialize = state is None or state_offset != offset
    if initialize:
        state = torch.empty(
            (_STATE_ROWS, _STATE_THREADS),
            device=out.device,
            dtype=torch.int64,
        )

    full_rounds = remaining // _STATE_THREADS
    if full_rounds:
        count = full_rounds * _STATE_THREADS
        _launch_raw_state(
            flat[written : written + count],
            state,
            seed,
            offset,
            full_rounds,
            initialize,
            rng_kind,
        )
        initialize = False
        written += count
        offset += count
        remaining -= count

    if remaining:
        pending = torch.empty(_STATE_THREADS, device=out.device, dtype=torch.int32)
        _launch_raw_state(pending, state, seed, offset, 1, initialize, rng_kind)
        flat[written:].copy_(pending[:remaining])
        generator._raw_sequence_pending = pending
        generator._raw_sequence_pending_offset = offset + remaining
        state_offset = offset + _STATE_THREADS
        offset += remaining
    else:
        state_offset = offset

    generator._raw_sequence_state = state
    generator._raw_sequence_state_offset = state_offset
    generator.offset = offset


def _consume_pending(out: torch.Tensor, generator, offset: int) -> int:
    pending = getattr(generator, "_raw_sequence_pending", None)
    pending_offset = int(getattr(generator, "_raw_sequence_pending_offset", -1))
    if pending is None or pending_offset != offset:
        if pending is not None:
            discard_state_prng_raw_sequence(generator)
        return 0

    state_offset = int(generator._raw_sequence_state_offset)
    available = state_offset - pending_offset
    take = min(out.numel(), available)
    start = pending.numel() - available
    out[:take].copy_(pending[start : start + take])
    if take == available:
        delattr(generator, "_raw_sequence_pending")
        delattr(generator, "_raw_sequence_pending_offset")
    else:
        generator._raw_sequence_pending_offset = pending_offset + take
    return take


def _launch_raw_state(
    out: torch.Tensor,
    state: torch.Tensor,
    seed: int,
    offset: int,
    num_iters: int,
    initialize: bool,
    rng_kind: int,
) -> None:
    _RAW_STATE_LAUNCHER.launch(
        (_STATE_THREADS // _BLOCK,),
        (
            out,
            state,
            seed & 0xFFFFFFFF,
            (seed >> 32) & 0xFFFFFFFF,
            offset & 0xFFFFFFFF,
            num_iters,
        ),
        (initialize, _STATE_THREADS, _BLOCK, rng_kind),
        (4,),
    )
