from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contract_benchmark.native_cuda_paths import (  # noqa: E402
    dependency_report,
    find_cuda_runtime_libraries,
    prepend_cuda_library_path,
    rpath_linker_flags,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optional legacy cuRAND Device API extension.")
    parser.add_argument("--verbose", action="store_true")
    default_build_dir = Path(
        os.environ.get(
            "CURAND_CONTRACT_DEVICE_BUILD_DIR",
            Path(__file__).resolve().parent / "build",
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
    ld_library_path = prepend_cuda_library_path()
    from torch.utils.cpp_extension import load

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    module = load(
        name="curand_contract_device_ext",
        sources=[str(root / "curand_contract_device_ext.cu")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=rpath_linker_flags(),
        build_directory=str(args.build_dir),
        verbose=args.verbose,
    )
    print(f"built {module.__name__}")
    print(f"module_file={getattr(module, '__file__', '')}")
    print(f"build_dir={args.build_dir}")
    print(f"ld_library_path={ld_library_path}")
    print(f"cudart_candidates={[str(path) for path in find_cuda_runtime_libraries()]}")
    module_file = getattr(module, "__file__", None)
    if module_file:
        print(f"dependency_report={dependency_report(Path(module_file))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
