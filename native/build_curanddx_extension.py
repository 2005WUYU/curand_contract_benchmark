from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from torch.utils.cpp_extension import load


HEADER_NAMES = (
    "curanddx.hpp",
    "curanddx/curanddx.hpp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optional cuRANDDx benchmark extension.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--sm",
        type=int,
        default=_default_sm(),
        help="cuRANDDx descriptor SM target, e.g. 900 for H20/Hopper.",
    )
    default_build_dir = Path(
        os.environ.get(
            "CURAND_CONTRACT_CURANDDX_BUILD_DIR",
            Path(__file__).resolve().parent / "build_curanddx",
        )
    )
    parser.add_argument("--build-dir", type=Path, default=default_build_dir)
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Reuse an existing build directory instead of forcing a rebuild.",
    )
    parser.set_defaults(clean=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    include_dirs = _candidate_include_dirs()
    header_paths = _find_headers(include_dirs)
    if not header_paths:
        raise SystemExit(
            "cuRANDDx headers were not found. Set MATHDX_ROOT or CPATH to the MathDx include tree; "
            f"searched: {[str(path) for path in include_dirs]}"
        )

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    module = load(
        name="curanddx_contract_ext",
        sources=[str(root / "curanddx_contract_ext.cu")],
        extra_include_paths=[str(path) for path in include_dirs if path.exists()],
        extra_cuda_cflags=["-O3", f"-DCURANDDX_TARGET_SM={args.sm}"],
        build_directory=str(args.build_dir),
        verbose=args.verbose,
    )
    print(f"built {module.__name__}")
    print(f"build_dir={args.build_dir}")
    print(f"curanddx_target_sm={args.sm}")
    print(f"curanddx_headers={[str(path) for path in header_paths]}")
    return 0


def _default_sm() -> int:
    env_value = os.environ.get("CURAND_CONTRACT_CURANDDX_SM")
    if env_value:
        return int(env_value)
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return major * 100 + minor * 10
    except Exception:
        pass
    return 900


def _candidate_include_dirs() -> list[Path]:
    candidates: list[Path] = []
    mathdx_root = os.environ.get("MATHDX_ROOT")
    if mathdx_root:
        root = Path(mathdx_root)
        candidates.extend([root / "include", root / "include" / "curanddx"])
    for env_name in ("CPATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH"):
        for part in os.environ.get(env_name, "").split(os.pathsep):
            if not part:
                continue
            path = Path(part)
            candidates.extend([path, path / "include", path / "include" / "curanddx"])
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
        if text in seen:
            continue
        seen.add(text)
        if path.exists():
            unique.append(path)
    return unique


def _find_headers(include_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for include_dir in include_dirs:
        for name in HEADER_NAMES:
            path = include_dir / name
            if path.exists() and str(path) not in seen:
                seen.add(str(path))
                found.append(path)
    return found


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
