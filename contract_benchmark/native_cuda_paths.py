from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
import subprocess
from typing import Any


CUDA_RUNTIME_NAMES = (
    "libcudart.so.12",
    "libcudart.so.13",
    "libcudart.so",
)


def cuda_library_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("CURAND_CONTRACT_CUDA_LIB_DIRS", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        candidates.extend(Path(part) for part in os.environ.get(env_name, "").split(os.pathsep) if part)
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_name)
        if root:
            candidates.extend([Path(root) / "lib64", Path(root) / "lib"])
    candidates.extend(
        [
            Path("/usr/local/cuda/lib64"),
            Path("/usr/local/cuda/lib"),
            Path("/opt/conda/lib"),
        ]
    )
    candidates.extend(Path(path) for path in glob.glob("/usr/local/cuda-*/lib64"))
    candidates.extend(Path(path) for path in glob.glob("/usr/local/cuda-*/lib"))
    candidates.extend(Path(path) for path in glob.glob("/opt/conda/lib/python*/site-packages/nvidia/*/lib"))
    try:
        import torch

        candidates.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    return _dedupe_existing_dirs(candidates)


def find_cuda_runtime_libraries() -> list[Path]:
    libraries: list[Path] = []
    for directory in cuda_library_dirs():
        for name in CUDA_RUNTIME_NAMES:
            path = directory / name
            if path.exists():
                libraries.append(path)
        libraries.extend(Path(path) for path in glob.glob(str(directory / "libcudart.so*")))
    return _dedupe_paths(libraries)


def prepend_cuda_library_path() -> str:
    existing = [Path(part) for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    parts = _dedupe_paths(cuda_library_dirs() + existing)
    text = os.pathsep.join(str(path) for path in parts)
    if text:
        os.environ["LD_LIBRARY_PATH"] = text
    return text


def rpath_linker_flags() -> list[str]:
    flags = ["-Wl,--enable-new-dtags"]
    for directory in cuda_library_dirs():
        flags.append(f"-Wl,-rpath,{directory}")
    return flags


def preload_cuda_runtime() -> dict[str, Any]:
    errors: list[str] = []
    preloaded: list[str] = []
    for library in find_cuda_runtime_libraries():
        try:
            ctypes.CDLL(str(library), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            preloaded.append(str(library))
        except OSError as exc:
            errors.append(f"{library}: {exc}")
    return {"preloaded": preloaded, "errors": errors}


def shared_object_candidates(build_dirs: list[Path], module_name: str) -> list[Path]:
    candidates: list[Path] = []
    for build_dir in build_dirs:
        candidates.extend(Path(path) for path in glob.glob(str(build_dir / f"{module_name}*.so")))
    return _dedupe_paths(candidates)


def dependency_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path)}
    try:
        completed = subprocess.run(
            ["ldd", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        report["error"] = str(exc)
        return report
    output = completed.stdout.strip()
    report["returncode"] = completed.returncode
    report["stdout"] = output
    report["missing"] = _missing_dependencies(output)
    return report


def native_extension_diagnostics(build_dirs: list[Path], module_name: str) -> dict[str, Any]:
    shared_objects = shared_object_candidates(build_dirs, module_name)
    reports = [dependency_report(path) for path in shared_objects]
    missing = sorted({dep for report in reports for dep in (report.get("missing") or [])})
    return {
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "cuda_library_dirs": [str(path) for path in cuda_library_dirs()],
        "cudart_candidates": [str(path) for path in find_cuda_runtime_libraries()],
        "shared_objects": [str(path) for path in shared_objects],
        "dependency_reports": reports,
        "missing_dependencies": missing,
    }


def _missing_dependencies(ldd_output: str) -> list[str]:
    missing: list[str] = []
    for line in ldd_output.splitlines():
        stripped = line.strip()
        if "not found" not in stripped:
            continue
        missing.append(stripped.split()[0])
    return missing


def _dedupe_existing_dirs(paths: list[Path]) -> list[Path]:
    return _dedupe_paths([path for path in paths if path.exists() and path.is_dir()])


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        unique.append(path)
    return unique
