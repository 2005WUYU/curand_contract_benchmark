from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


REQUIRED_SYMBOLS = (
    "philox_raw_u32",
    "philox_uniform",
    "philox_add_uniform",
    "philox_threshold",
    "philox_dropout",
)


def find_built_curanddx_extension() -> tuple[Any | None, str | None]:
    root = Path(__file__).resolve().parents[1]
    build_dirs = _build_dirs(root)
    _prepend_build_dirs(build_dirs)
    try:
        import curanddx_contract_ext  # type: ignore

        missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(curanddx_contract_ext, symbol)]
        if missing:
            return None, f"cuRANDDx extension imported but is missing required symbols: {missing}"
        return curanddx_contract_ext, None
    except Exception as exc:
        expected = root / "native" / "curanddx_contract_ext.cu"
        return None, (
            "cuRANDDx extension is not importable. Build it on H20/Linux via "
            "native/build_curanddx_extension.py. "
            f"Import error: {exc}. Expected source near {expected}; "
            f"searched build dirs: {[str(path) for path in build_dirs]}."
        )


def _build_dirs(root: Path) -> list[Path]:
    if os.environ.get("CURAND_CONTRACT_CURANDDX_BUILD_DIR"):
        return [Path(os.environ["CURAND_CONTRACT_CURANDDX_BUILD_DIR"])]
    return [root / "native" / "build_curanddx"]


def _prepend_build_dirs(build_dirs: list[Path]) -> None:
    for build_dir in reversed(build_dirs):
        text = str(build_dir)
        if not build_dir.exists():
            continue
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
