from __future__ import annotations

import os
from pathlib import Path
from typing import Any


HEADER_NAMES = (
    "curanddx.hpp",
    "curanddx/curanddx.hpp",
)


def curanddx_status() -> dict[str, Any]:
    include_dirs = _candidate_include_dirs()
    headers = _find_headers(include_dirs)
    module, extension_reason = _load_extension()
    status: dict[str, Any] = {
        "available": bool(headers) and module is not None,
        "headers_available": bool(headers),
        "header_paths": [str(path) for path in headers],
        "mathdx_root": os.environ.get("MATHDX_ROOT"),
        "include_dirs_checked": [str(path) for path in include_dirs],
        "extension_available": module is not None,
        "extension_build_dir": os.environ.get("CURAND_CONTRACT_CURANDDX_BUILD_DIR"),
    }
    if module is not None:
        status["extension_symbols"] = _extension_symbols(module)
        status["unsupported_reason"] = None
    elif headers:
        status["unsupported_reason"] = extension_reason or (
            "cuRANDDx headers were found, but the cuRANDDx benchmark extension "
            "is not built or not importable."
        )
    else:
        status["unsupported_reason"] = (
            "cuRANDDx headers were not found. Run inside the MathDx/cuRANDDx "
            "container and set MATHDX_ROOT or CPATH to the MathDx include tree."
        )
        if extension_reason:
            status["extension_import_error"] = extension_reason
    return status


def _load_extension() -> tuple[Any | None, str | None]:
    try:
        from contract_benchmark.optional_curanddx_api import find_built_curanddx_extension
    except Exception as exc:
        return None, f"optional cuRANDDx loader import failed: {exc}"
    return find_built_curanddx_extension()


def _extension_symbols(module: Any) -> dict[str, bool]:
    try:
        from contract_benchmark.optional_curanddx_api import REQUIRED_SYMBOLS
    except Exception:
        REQUIRED_SYMBOLS = ()
    return {symbol: hasattr(module, symbol) for symbol in REQUIRED_SYMBOLS}


def _candidate_include_dirs() -> list[Path]:
    candidates: list[Path] = []
    mathdx_root = os.environ.get("MATHDX_ROOT")
    if mathdx_root:
        root = Path(mathdx_root)
        candidates.extend(
            [
                root / "include",
                root / "include" / "curanddx",
            ]
        )
    for env_name in ("CPATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH"):
        for part in os.environ.get(env_name, "").split(os.pathsep):
            if part:
                path = Path(part)
                candidates.append(path)
                candidates.append(path / "include")
                candidates.append(path / "include" / "curanddx")
    candidates.extend(
        [
            Path("/opt/mathdx/current/include"),
            Path("/opt/mathdx/current/include/curanddx"),
            Path("/usr/local/cuda/include"),
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        text = str(path)
        if text not in seen:
            seen.add(text)
            unique.append(path)
    return unique


def _find_headers(include_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for include_dir in include_dirs:
        for name in HEADER_NAMES:
            path = include_dir / name
            if path.exists():
                text = str(path)
                if text not in seen:
                    seen.add(text)
                    found.append(path)
    return found
