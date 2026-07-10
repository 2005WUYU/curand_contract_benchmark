from __future__ import annotations

import torch

from flagrand.rng._mt19937_data import MT19937_N, MT19937_NUM_STREAMS, MT19937_SEQUENCE_CHUNK
from flagrand.rng._mt19937_stream_kernel import launch_mt19937_stream
from flagrand.rng._stateful_output import StatefulOutput


def generate_streamed_mt19937(
    generator,
    out: torch.Tensor,
    offset_val: int,
    num_warps: int,
    output: StatefulOutput,
) -> None:
    flat = out if out.ndim == 1 else out.view(-1)
    written = 0
    current = int(offset_val)
    remaining = flat.numel()

    while remaining:
        prefix_start = generator._mt19937_prefix_start
        prefix_available = generator._mt19937_prefix_count
        if prefix_available and prefix_start != current:
            raise RuntimeError(
                "MT19937 raw prefix cache is not aligned with the generator offset: "
                f"cache_start={prefix_start}, offset={current}."
            )

        prefix_count = min(remaining, prefix_available)
        generated_elements = min(remaining - prefix_count, MT19937_SEQUENCE_CHUNK)
        block_count = (generated_elements + MT19937_N - 1) // MT19937_N
        segment_elements = prefix_count + generated_elements
        segment = (
            flat
            if written == 0 and segment_elements == flat.numel()
            else flat[written : written + segment_elements]
        )
        start_stream = (generator._ws_next_block_start // MT19937_N) % MT19937_NUM_STREAMS
        launch_mt19937_stream(
            generator,
            segment,
            prefix_offset=generator._mt19937_prefix_offset,
            prefix_count=prefix_count,
            block_count=block_count,
            start_stream=start_stream,
            output=output,
            num_warps=num_warps,
        )

        written += segment_elements
        current += segment_elements
        remaining -= segment_elements
        if block_count == 0:
            generator._mt19937_prefix_offset += prefix_count
            generator._mt19937_prefix_count -= prefix_count
            generator._mt19937_prefix_start = current
            continue

        generator._mt19937_prefix_raw, generator._mt19937_next_raw = (
            generator._mt19937_next_raw,
            generator._mt19937_prefix_raw,
        )
        used_in_last_block = generated_elements - (block_count - 1) * MT19937_N
        generator._mt19937_prefix_offset = (
            0 if used_in_last_block == MT19937_N else used_in_last_block
        )
        generator._mt19937_prefix_count = MT19937_N - used_in_last_block
        generator._mt19937_prefix_start = current
        generator._ws_next_block_start += block_count * MT19937_N
