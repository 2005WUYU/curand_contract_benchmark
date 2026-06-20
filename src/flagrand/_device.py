from __future__ import annotations

import torch

SUPPORTED_DEVICE_TYPES: frozenset[str] = frozenset({"cuda"})


def is_accelerator_available() -> bool:
    if "cuda" in SUPPORTED_DEVICE_TYPES:
        return torch.cuda.is_available()
    return False


def require_accelerator() -> torch.device:
    if not is_accelerator_available():
        raise RuntimeError(
            "No accelerator device supported by the current build was detected. "
            "Please verify that drivers, runtime, and PyTorch installation are compatible."
        )
    return default_accelerator()


def default_accelerator() -> torch.device:
    if "cuda" in SUPPORTED_DEVICE_TYPES and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    raise RuntimeError("Internal error: default accelerator device requested while unavailable.")


def assert_tensor_device_supported(tensor: torch.Tensor, *, op_name: str) -> None:
    if tensor.device.type not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
        raise ValueError(
            f"{op_name}: output must be on a supported device "
            f"(type={tensor.device.type!r}, supported: {supported})."
        )


def synchronize_accelerator(device: torch.device | None = None) -> None:
    dev = device if device is not None else default_accelerator()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        return
    raise NotImplementedError(
        f"synchronize_accelerator: no synchronization implementation provided for device type {dev.type!r}."
    )
