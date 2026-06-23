from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any

from contract_benchmark.native_cuda_paths import (
    native_extension_diagnostics,
    preload_cuda_runtime,
    prepend_cuda_library_path,
)


REQUIRED_SYMBOLS = (
    "philox_raw_u32",
    "philox_uniform",
    "philox_add_uniform",
    "philox_threshold",
    "philox_dropout",
)


@lru_cache(maxsize=1)
def find_built_curand_device_extension() -> tuple[Any | None, str | None]:
    root = Path(__file__).resolve().parents[1]
    build_dirs = _build_dirs(root)
    _prepend_build_dirs(build_dirs)
    preload_info = _prepare_cuda_runtime()
    try:
        import curand_contract_device_ext  # type: ignore

        missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(curand_contract_device_ext, symbol)]
        if missing:
            return None, f"legacy cuRAND Device API extension imported but is missing required symbols: {missing}"
        return curand_contract_device_ext, None
    except Exception as exc:
        expected = root / "native" / "curand_contract_device_ext"
        diagnostics = native_extension_diagnostics(build_dirs, "curand_contract_device_ext")
        return None, (
            "legacy cuRAND Device API extension is not importable. "
            f"Build it on H20/Linux via native/build_curand_device_extension.py. "
            f"Import error: {exc}. Expected source/build root near {expected}; "
            f"searched build dirs: {[str(path) for path in build_dirs]}. "
            f"preload: {preload_info}. "
            f"missing_dependencies: {diagnostics.get('missing_dependencies')}. "
            f"cudart_candidates: {diagnostics.get('cudart_candidates')}."
        )


def curand_device_extension_status() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    build_dirs = _build_dirs(root)
    module, reason = find_built_curand_device_extension()
    diagnostics = native_extension_diagnostics(build_dirs, "curand_contract_device_ext")
    status: dict[str, Any] = {
        "available": module is not None,
        "unsupported_reason": reason if module is None else None,
        "build_dirs": [str(path) for path in build_dirs],
        "ld_library_path": diagnostics.get("ld_library_path"),
        "cuda_library_dirs": diagnostics.get("cuda_library_dirs"),
        "cudart_candidates": diagnostics.get("cudart_candidates"),
        "shared_objects": diagnostics.get("shared_objects"),
        "missing_dependencies": diagnostics.get("missing_dependencies"),
    }
    if module is not None:
        status["module_file"] = getattr(module, "__file__", None)
        status["extension_symbols"] = {symbol: hasattr(module, symbol) for symbol in REQUIRED_SYMBOLS}
    else:
        status["dependency_reports"] = diagnostics.get("dependency_reports")
    return status


def _build_dirs(root: Path) -> list[Path]:
    if os.environ.get("CURAND_CONTRACT_DEVICE_BUILD_DIR"):
        return [Path(os.environ["CURAND_CONTRACT_DEVICE_BUILD_DIR"])]
    return [root / "native" / "build"]


def _prepend_build_dirs(build_dirs: list[Path]) -> None:
    for build_dir in reversed(build_dirs):
        text = str(build_dir)
        if not build_dir.exists():
            continue
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _prepare_cuda_runtime() -> dict[str, Any]:
    ld_library_path = prepend_cuda_library_path()
    preload_info = preload_cuda_runtime()
    preload_info["ld_library_path"] = ld_library_path
    return preload_info
