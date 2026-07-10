from __future__ import annotations

import torch

from flagrand._device import assert_tensor_device_supported, require_accelerator


def prepare_quasi_output(
    out: torch.Tensor,
    *,
    dtype: torch.dtype,
    dimensions: int,
    max_dimensions: int,
    offset: int,
    op_name: str,
) -> None:
    require_accelerator()
    if out.dtype != dtype:
        raise TypeError(f"{op_name} requires {dtype} output.")
    assert_tensor_device_supported(out, op_name=op_name)
    if dimensions < 1 or dimensions > max_dimensions:
        raise ValueError(
            f"{op_name}: dimensions must be between 1 and {max_dimensions}, got {dimensions}."
        )
    if offset < 0:
        raise ValueError(f"{op_name}: offset must be >= 0, got {offset}.")
    if out.numel() % dimensions != 0:
        raise ValueError(f"{op_name}: element count must be a multiple of dimensions.")


def offset_from_kwargs(default: int, kwargs: dict[str, object]) -> int:
    return int(kwargs.get("offset", default))


def normal_params(kwargs: dict[str, object]) -> tuple[float, float]:
    return float(kwargs.get("mean", 0.0)), float(kwargs.get("stddev", 1.0))
