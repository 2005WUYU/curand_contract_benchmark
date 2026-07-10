from __future__ import annotations

from typing import Any
import weakref

import torch
from triton.runtime.driver import driver


def find_matching_entry(
    entries: list[Any],
    values: tuple[object, ...],
    option_names: tuple[str, ...],
    option_values: tuple[object, ...],
) -> Any | None:
    for entry in entries:
        if _matches_hot_entry(entry, values, option_names, option_values):
            return entry

    device = _argument_device(values)
    candidates = [entry for entry in entries if entry.device == device]
    if not candidates:
        return None
    specialization, parsed_options = bind_specialization(
        candidates[0].binder,
        values,
        option_names,
        option_values,
    )
    for entry in candidates:
        if entry.specialization == specialization and entry.parsed_options == parsed_options:
            entry.value_tokens = value_tokens(values)
            return entry
    return None


def bind_specialization(
    binder: Any,
    values: tuple[object, ...],
    option_names: tuple[str, ...],
    option_values: tuple[object, ...],
) -> tuple[tuple[object, ...], object]:
    if option_names == ("num_warps",):
        _, specialization, parsed_options = binder(
            *values,
            num_warps=option_values[0],
        )
    else:
        options = dict(zip(option_names, option_values))
        _, specialization, parsed_options = binder(*values, **options)
    return tuple(specialization), parsed_options


def value_tokens(values: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        ("tensor", weakref.ref(value), value.data_ptr())
        if isinstance(value, torch.Tensor)
        else ("value", value)
        for value in values
    )


def _matches_hot_entry(
    entry: Any,
    values: tuple[object, ...],
    option_names: tuple[str, ...],
    option_values: tuple[object, ...],
) -> bool:
    if len(values) != len(entry.value_tokens):
        return False
    if option_names != ("num_warps",) or not isinstance(entry.parsed_options, dict):
        return False
    if entry.parsed_options.get("num_warps") != option_values[0]:
        return False
    for value, token, component in zip(values, entry.value_tokens, entry.specialization):
        if token[0] == "tensor":
            if token[1]() is not value or value.data_ptr() != token[2]:
                return False
            continue
        if value == token[1]:
            continue
        if not _matches_changed_value(value, component):
            return False
    return True


def _matches_changed_value(value: object, component: tuple[object, ...]) -> bool:
    signature = component[0]
    if signature == "i32":
        matches_type = isinstance(value, int) and -(1 << 31) <= value < (1 << 31)
    elif signature == "u32":
        matches_type = isinstance(value, int) and 0 <= value < (1 << 32)
    elif signature == "i64":
        matches_type = isinstance(value, int) and -(1 << 63) <= value < (1 << 63)
    elif signature == "u64":
        matches_type = isinstance(value, int) and 0 <= value < (1 << 64)
    elif signature in {"fp32", "fp64"}:
        return isinstance(value, float)
    else:
        return False
    if not matches_type or isinstance(value, bool):
        return False
    return len(component) < 2 or component[1] != "D" or value % 16 == 0


def _argument_device(values: tuple[object, ...]) -> int:
    for value in values:
        if isinstance(value, torch.Tensor) and value.device.index is not None:
            return value.device.index
    return driver.active.get_current_device()
