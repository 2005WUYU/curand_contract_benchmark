from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optional legacy cuRAND Device API extension.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parent / "build")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    args.build_dir.mkdir(parents=True, exist_ok=True)
    module = load(
        name="curand_contract_device_ext",
        sources=[str(root / "curand_contract_device_ext.cu")],
        extra_cuda_cflags=["-O3"],
        build_directory=str(args.build_dir),
        verbose=args.verbose,
    )
    print(f"built {module.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

