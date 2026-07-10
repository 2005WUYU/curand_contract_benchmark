from __future__ import annotations

import torch

_DEVICE_TABLES: dict[tuple[int, tuple[int, ...], tuple[int, ...], str, torch.dtype], torch.Tensor] = {}
_CHUNK_TABLES: dict[tuple[int, tuple[int, ...], int, int, str, torch.dtype], torch.Tensor] = {}
_EMPTY_DEVICE_TABLES: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_LAUNCH_PLANS: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, int, int]] = {}


def launch_plan32(out, direction_vectors, scramble_constants, dimensions, points, offset, block):
    return launch_plan(
        direction_vectors,
        scramble_constants,
        dimensions=dimensions,
        points_per_dim=points,
        offset=offset,
        max_bits=32,
        block_size=block,
        device=str(out.device),
        dtype=torch.int32,
    )


def launch_plan64(out, direction_vectors, scramble_constants, dimensions, points, offset, block):
    return launch_plan(
        direction_vectors,
        scramble_constants,
        dimensions=dimensions,
        points_per_dim=points,
        offset=offset,
        max_bits=64,
        block_size=block,
        device=str(out.device),
        dtype=torch.int64,
    )


def launch_plan(
    direction_vectors: torch.Tensor,
    scramble_constants: torch.Tensor | None,
    *,
    dimensions: int,
    points_per_dim: int,
    offset: int,
    max_bits: int,
    block_size: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    key = (
        direction_vectors.data_ptr(),
        0 if scramble_constants is None else scramble_constants.data_ptr(),
        dimensions,
        points_per_dim,
        offset,
        max_bits,
        block_size,
        device,
        dtype,
    )
    cached = _LAUNCH_PLANS.get(key)
    if cached is None:
        chunks = _ceil_div(required_bits(points_per_dim, offset, max_bits), 8)
        cached = (
            chunk_table(direction_vectors, dimensions, chunks, device, dtype),
            optional_device_table(scramble_constants, device, dtype),
            chunks,
            _ceil_div(points_per_dim, block_size),
        )
        _LAUNCH_PLANS[key] = cached
    return cached


def optional_device_table(
    tensor: torch.Tensor | None, device: str, dtype: torch.dtype
) -> torch.Tensor:
    if tensor is None:
        key = (device, dtype)
        cached = _EMPTY_DEVICE_TABLES.get(key)
        if cached is None:
            cached = torch.empty(1, device=device, dtype=dtype)
            _EMPTY_DEVICE_TABLES[key] = cached
        return cached
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


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
