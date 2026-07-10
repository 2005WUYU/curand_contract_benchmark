from __future__ import annotations

import torch

from flagrand._device import require_accelerator, assert_tensor_device_supported
from flagrand.fused._internal.utils import (
    get_generator_type,
    GENERATOR_PHILOX,
    GENERATOR_SOBOL64,
    GENERATOR_SCRAMBLED_SOBOL64,
)

_64BIT_GENERATORS = {GENERATOR_SOBOL64, GENERATOR_SCRAMBLED_SOBOL64}


def generate_raw(
    out: torch.Tensor,
    generator,
    *,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    require_accelerator()

    gen_type = get_generator_type(generator)
    is_64 = gen_type in _64BIT_GENERATORS

    if is_64:
        if out.dtype != torch.int64:
            raise TypeError("generate_raw: int64 output required for Generator64.")
    else:
        if out.dtype == torch.int64:
            raise TypeError(
                "generate_raw: int64 raw output is supported only for Sobol64 "
                f"and ScrambledSobol64 generators, got {type(generator).__name__}."
            )
        if out.dtype != torch.int32:
            raise TypeError("generate_raw: int32 output required for Generator32.")

    assert_tensor_device_supported(out, op_name="generate_raw")

    n = out.numel()
    if n == 0:
        return out
    if block_size is not None and block_size <= 0:
        raise ValueError(f"generate_raw: block_size must be > 0, got {block_size}.")
    if num_warps is not None and num_warps <= 0:
        raise ValueError(f"generate_raw: num_warps must be > 0, got {num_warps}.")

    if not is_64:
        if gen_type == GENERATOR_PHILOX:
            if n % 4 != 0:
                raise ValueError(
                    f"generate_raw: Philox requires element count to be "
                    f"a multiple of 4, got {n}."
                )

    launch_kwargs: dict[str, int] = {}
    if block_size is not None:
        launch_kwargs["block_size"] = block_size
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    if is_64:
        return generator.generate_long_long(out, **launch_kwargs)
    else:
        return generator.generate(out, **launch_kwargs)
