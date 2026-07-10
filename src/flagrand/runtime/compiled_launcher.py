from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triton import knobs
from triton.runtime.driver import driver

from flagrand.runtime.specialization import (
    Constraint,
    build_constraints,
    matches_constraints,
    runtime_positions,
)


@dataclass(frozen=True)
class _LaunchEntry:
    compiled: Any
    device: int
    constexpr_values: tuple[object, ...]
    option_values: tuple[object, ...]
    constraints: tuple[Constraint, ...]
    runtime_positions: tuple[int, ...]
    launch_metadata: object

    def matches(
        self,
        values: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
    ) -> bool:
        return bool(
            self.constexpr_values == constexpr_values
            and self.option_values == option_values
            and matches_constraints(values, self.constraints)
        )


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

    def launch(
        self,
        grid: int | tuple[int, ...],
        args: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
    ) -> Any:
        grid3 = _normalize_grid(grid)
        values = args + constexpr_values
        if not _requires_regular_jit(self._kernel):
            for entry in self._entries:
                if entry.matches(values, constexpr_values, option_values):
                    return _launch_entry(entry, grid3, values)
        return self._compile_and_launch(
            grid3,
            args,
            values,
            constexpr_values,
            option_values,
        )

    def _compile_and_launch(
        self,
        grid3: tuple[int, int, int],
        args: tuple[object, ...],
        values: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
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

        positions = runtime_positions(compiled)
        runtime_args = tuple(values[index] for index in positions)
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        compiled._init_handles()
        entry = _LaunchEntry(
            compiled=compiled,
            device=device,
            constexpr_values=constexpr_values,
            option_values=option_values,
            constraints=build_constraints(compiled, values),
            runtime_positions=positions,
            launch_metadata=compiled.launch_metadata(grid3, stream, *runtime_args),
        )
        self._entries.insert(0, entry)
        del self._entries[self._max_entries :]
        return compiled


def _launch_entry(
    entry: _LaunchEntry,
    grid: tuple[int, int, int],
    values: tuple[object, ...],
) -> Any:
    compiled = entry.compiled
    stream = driver.active.get_current_stream(entry.device)
    runtime_args = tuple(values[index] for index in entry.runtime_positions)
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
        *runtime_args,
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
