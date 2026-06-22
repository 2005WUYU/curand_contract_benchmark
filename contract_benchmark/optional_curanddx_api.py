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
    build_dirs = []
    if os.environ.get("CURAND_CONTRACT_CURANDDX_BUILD_DIR"):
        build_dirs.append(Path(os.environ["CURAND_CONTRACT_CURANDDX_BUILD_DIR"]))
    build_dirs.append(root / "native" / "build_curanddx")
    for build_dir in build_dirs:
        if build_dir.exists() and str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
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
