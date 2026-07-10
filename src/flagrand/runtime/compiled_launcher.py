from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from triton import knobs
from triton.runtime.driver import driver

from flagrand.runtime.specialization import (
    bind_specialization,
    find_matching_entry,
    value_tokens,
)


@dataclass
class _LaunchEntry:
    compiled: Any
    device: int
    binder: Any
    specialization: tuple[object, ...]
    parsed_options: object
    value_tokens: tuple[tuple[object, ...], ...]
    launch_metadata: object


class CachedKernelLauncher:
    """Reuse a compiled Triton specialization while resolving each call's stream."""

    def __init__(
        self,
        kernel: Any,
        *,
        constexpr_names: tuple[str, ...],
        option_names: tuple[str, ...] = ("num_warps",),
        max_entries: int = 32,
    ) -> None:
        self._kernel = kernel
        self._constexpr_names = constexpr_names
        self._option_names = option_names
        self._max_entries = max_entries
        self._entries: list[_LaunchEntry] = []
        self._keyed_entries: OrderedDict[object, _LaunchEntry] = OrderedDict()

    def launch(
        self,
        grid: int | tuple[int, ...],
        args: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
        *,
        specialization_key: object | None = None,
    ) -> Any:
        grid3 = _normalize_grid(grid)
        values = args + constexpr_values
        if not _requires_regular_jit(self._kernel):
            if specialization_key is not None:
                entry = self._keyed_entries.get(specialization_key)
                if entry is not None:
                    self._keyed_entries.move_to_end(specialization_key)
                    return _launch_entry(entry, grid3, values)
            entry = find_matching_entry(
                self._entries,
                values,
                self._option_names,
                option_values,
            )
            if entry is not None:
                self._remember_key(specialization_key, entry)
                return _launch_entry(entry, grid3, values)
        return self._compile_and_launch(
            grid3,
            args,
            values,
            constexpr_values,
            option_values,
            specialization_key,
        )

    def _compile_and_launch(
        self,
        grid3: tuple[int, int, int],
        args: tuple[object, ...],
        values: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
        specialization_key: object | None,
    ) -> Any:
        if len(constexpr_values) != len(self._constexpr_names):
            raise ValueError("constexpr value count does not match launcher configuration")
        if len(option_values) != len(self._option_names):
            raise ValueError("option value count does not match launcher configuration")

        kwargs = dict(zip(self._constexpr_names, constexpr_values))
        kwargs.update(zip(self._option_names, option_values))
        compiled = self._kernel[grid3](*args, **kwargs)
        if compiled is None or _requires_regular_jit(self._kernel):
            return compiled

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        compiled._init_handles()
        binder = self._kernel.device_caches[device][4]
        specialization, parsed_options = bind_specialization(
            binder,
            values,
            self._option_names,
            option_values,
        )
        entry = _LaunchEntry(
            compiled=compiled,
            device=device,
            binder=binder,
            specialization=specialization,
            parsed_options=parsed_options,
            value_tokens=value_tokens(values),
            launch_metadata=compiled.launch_metadata(grid3, stream, *values),
        )
        self._entries.insert(0, entry)
        del self._entries[self._max_entries :]
        self._remember_key(specialization_key, entry)
        return compiled

    def _remember_key(self, key: object | None, entry: _LaunchEntry) -> None:
        if key is None:
            return
        self._keyed_entries[key] = entry
        self._keyed_entries.move_to_end(key)
        while len(self._keyed_entries) > self._max_entries:
            self._keyed_entries.popitem(last=False)


def _launch_entry(
    entry: _LaunchEntry,
    grid: tuple[int, int, int],
    values: tuple[object, ...],
) -> Any:
    compiled = entry.compiled
    stream = driver.active.get_current_stream(entry.device)
    compiled.run(
        grid[0],
        grid[1],
        grid[2],
        stream,
        compiled.function,
        compiled.packed_metadata,
        entry.launch_metadata,
        None,
        None,
        *values,
    )
    return compiled


def _requires_regular_jit(kernel: Any) -> bool:
    return bool(
        kernel.pre_run_hooks
        or getattr(knobs.runtime.launch_enter_hook, "calls", ())
        or getattr(knobs.runtime.launch_exit_hook, "calls", ())
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.add_stages_inspection_hook is not None
        or kernel.launch_metadata is not None
    )


def _normalize_grid(grid: int | tuple[int, ...]) -> tuple[int, int, int]:
    if isinstance(grid, int):
        return (grid, 1, 1)
    if not 1 <= len(grid) <= 3:
        raise ValueError(f"Triton grid must have one to three dimensions, got {grid!r}")
    return (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
