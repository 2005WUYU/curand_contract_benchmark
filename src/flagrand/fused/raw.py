from __future__ import annotations

import torch

from flagrand._device import require_accelerator, assert_tensor_device_supported
from flagrand.fused._internal.utils import (
    get_generator_type,
    GENERATOR_PHILOX,
    GENERATOR_MTGP32,
    GENERATOR_SOBOL64,
    GENERATOR_SCRAMBLED_SOBOL64,
)

_64BIT_GENERATORS = {GENERATOR_SOBOL64, GENERATOR_SCRAMBLED_SOBOL64}


def generate_raw(
    out: torch.Tensor,
    generator,
) -> torch.Tensor:
    require_accelerator()

    gen_type = get_generator_type(generator)
    is_64 = gen_type in _64BIT_GENERATORS

    if is_64:
        if out.dtype != torch.int64:
            raise TypeError("generate_raw: int64 output required for Generator64.")
    else:
        if out.dtype != torch.int32:
            raise TypeError("generate_raw: int32 output required for Generator32.")

    assert_tensor_device_supported(out, op_name="generate_raw")

    n = out.numel()
    if n == 0:
        return out

    if not is_64:
        if gen_type == GENERATOR_PHILOX:
            if n % 4 != 0:
                raise ValueError(
                    f"generate_raw: Philox requires element count to be "
                    f"a multiple of 4, got {n}."
                )
        elif gen_type == GENERATOR_MTGP32:
            if generator.offset != 0:
                raise ValueError(
                    f"generate_raw: MTGP32 does not support non-zero offset, "
                    f"got {generator.offset}."
                )

    if is_64:
        return generator.generate_long_long(out)
    else:
        return generator.generate(out)

