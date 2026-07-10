from __future__ import annotations

import torch

_DEVICE_TABLES: dict[tuple[int, tuple[int, ...], tuple[int, ...], str, torch.dtype], torch.Tensor] = {}
_CHUNK_TABLES: dict[tuple[int, tuple[int, ...], int, int, str, torch.dtype], torch.Tensor] = {}


def optional_device_table(
    tensor: torch.Tensor | None, device: str, dtype: torch.dtype
) -> torch.Tensor:
    if tensor is None:
        return torch.empty(1, device=device, dtype=dtype)
    return _device_table(tensor, device, dtype)


def chunk_table(
    direction_vectors: torch.Tensor,
    dimensions: int,
    chunks: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (direction_vectors.data_ptr(), tuple(direction_vectors.shape), dimensions, chunks, device, dtype)
    cached = _CHUNK_TABLES.get(key)
    if cached is None:
        cached = _build_chunk_table(direction_vectors, dimensions, chunks, dtype).to(device)
        _CHUNK_TABLES[key] = cached
    return cached


def required_bits(points_per_dim: int, offset: int, max_bits: int) -> int:
    max_index = max(0, int(points_per_dim) + int(offset) - 1)
    return max(1, min(max_bits, max_index.bit_length()))


def _device_table(tensor: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), device, dtype)
    cached = _DEVICE_TABLES.get(key)
    if cached is None:
        cached = tensor.contiguous().view(dtype).to(device)
        _DEVICE_TABLES[key] = cached
    return cached


def _build_chunk_table(
    direction_vectors: torch.Tensor,
    dimensions: int,
    chunks: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    values: list[int] = []
    cols = direction_vectors.shape[1]
    for dim in range(dimensions):
        row = direction_vectors[dim]
        for chunk in range(chunks):
            base_bit = chunk * 8
            for byte in range(256):
                acc = 0
                for bit in range(8):
                    col = base_bit + bit
                    if col < cols and (byte & (1 << bit)):
                        acc ^= int(row[col].item())
                values.append(acc)
    unsigned_dtype = torch.uint64 if dtype is torch.int64 else torch.uint32
    return torch.tensor(values, dtype=unsigned_dtype).view(dtype).contiguous()
