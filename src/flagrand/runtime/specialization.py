from __future__ import annotations

import torch


Constraint = tuple[object, ...]

_TENSOR = 0
_INTEGER = 1
_FLOAT = 2
_EXACT = 3
_OTHER = 4


def build_constraints(compiled, values: tuple[object, ...]) -> tuple[Constraint, ...]:
    signature_items = tuple(compiled.src.signature.items())
    if len(signature_items) != len(values):
        raise RuntimeError("compiled Triton signature does not match launcher arguments")
    return tuple(
        _constraint_for(
            value,
            signature,
            compiled.src.attrs.get((index,), ()),
        )
        for index, (value, (_, signature)) in enumerate(zip(values, signature_items))
    )


def runtime_positions(compiled) -> tuple[int, ...]:
    return tuple(
        index
        for index, (_, signature) in enumerate(compiled.src.signature.items())
        if signature != "constexpr"
    )


def matches_constraints(
    values: tuple[object, ...],
    constraints: tuple[Constraint, ...],
) -> bool:
    for value, constraint in zip(values, constraints):
        kind = constraint[0]
        if kind == _EXACT:
            if value != constraint[1]:
                return False
            continue
        if kind == _TENSOR:
            if not isinstance(value, torch.Tensor):
                return False
            _, dtype, device_type, device_index, divisor = constraint
            if (
                value.dtype != dtype
                or value.device.type != device_type
                or value.device.index != device_index
                or (divisor and value.data_ptr() % int(divisor) != 0)
            ):
                return False
            continue
        if kind == _INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            _, signature, divisor = constraint
            if not _integer_fits(value, str(signature)) or (
                divisor and value % int(divisor) != 0
            ):
                return False
            continue
        if kind == _FLOAT:
            if not isinstance(value, float):
                return False
            continue
        if not isinstance(value, constraint[1]):
            return False
    return True


def _constraint_for(value: object, signature: str, attrs: object) -> Constraint:
    if signature == "constexpr":
        return (_EXACT, value)
    divisor = _divisibility(attrs)
    if isinstance(value, torch.Tensor):
        return (
            _TENSOR,
            value.dtype,
            value.device.type,
            value.device.index,
            divisor,
        )
    if isinstance(value, bool):
        return (_EXACT, value)
    if isinstance(value, int):
        return (_INTEGER, signature, divisor)
    if isinstance(value, float):
        return (_FLOAT, signature)
    return (_OTHER, type(value))


def _divisibility(attrs: object) -> int:
    for attr in attrs:
        if len(attr) == 2 and attr[0] == "tt.divisibility":
            return int(attr[1])
    return 0


def _integer_fits(value: int, signature: str) -> bool:
    if signature == "i32":
        return -(1 << 31) <= value < (1 << 31)
    if signature == "u32":
        return 0 <= value < (1 << 32)
    if signature == "i64":
        return -(1 << 63) <= value < (1 << 63)
    if signature == "u64":
        return 0 <= value < (1 << 64)
    return False
