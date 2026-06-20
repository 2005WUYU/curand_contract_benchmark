from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def find_built_curand_device_extension() -> tuple[Any | None, str | None]:
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "native" / "build"
    if build_dir.exists() and str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        import curand_contract_device_ext  # type: ignore

        return curand_contract_device_ext, None
    except Exception as exc:
        expected = root / "native" / "curand_contract_device_ext"
        return None, (
            "legacy cuRAND Device API extension is not importable. "
            f"Build it on H20/Linux via native/build_curand_device_extension.py. "
            f"Import error: {exc}. Expected source/build root near {expected}."
        )
