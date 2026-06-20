from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from contract_benchmark.curand_constants import CURAND_STATUS_SUCCESS


class CurandError(RuntimeError):
    def __init__(self, status: int, operation: str):
        self.status = int(status)
        self.operation = operation
        super().__init__(f"{operation} failed with cuRAND status {status}")


def _check(status: int, operation: str) -> None:
    if int(status) != CURAND_STATUS_SUCCESS:
        raise CurandError(int(status), operation)


def _add_windows_dll_dirs() -> None:
    if os.name != "nt":
        return
    candidates: list[Path] = []
    for key in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value) / "bin")
    path_value = os.environ.get("PATH", "")
    candidates.extend(Path(p) for p in path_value.split(os.pathsep) if p)
    for path in candidates:
        try:
            if path.exists():
                os.add_dll_directory(str(path))
        except (FileNotFoundError, OSError):
            continue


def _library_candidates() -> list[str]:
    if os.name == "nt":
        candidates = ["curand64_12.dll", "curand64_11.dll", "curand64_10.dll"]
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            candidates.extend(
                str(Path(cuda_path) / "bin" / name)
                for name in ("curand64_12.dll", "curand64_11.dll", "curand64_10.dll")
            )
        return candidates
    return ["libcurand.so", "libcurand.so.12", "libcurand.so.11", "libcurand.so.10"]


def _load_curand() -> tuple[ctypes.CDLL, str]:
    _add_windows_dll_dirs()
    errors: list[str] = []
    for candidate in _library_candidates():
        try:
            lib = ctypes.CDLL(candidate)
            return lib, candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Unable to load cuRAND shared library. Tried:\n{joined}")


_libcurand, CURAND_LIBRARY = _load_curand()


def _bind() -> None:
    lib = _libcurand
    gen_p = ctypes.POINTER(ctypes.c_void_p)
    size_t = ctypes.c_size_t
    ull = ctypes.c_ulonglong

    lib.curandCreateGenerator.argtypes = [gen_p, ctypes.c_int]
    lib.curandCreateGenerator.restype = ctypes.c_int
    lib.curandDestroyGenerator.argtypes = [ctypes.c_void_p]
    lib.curandDestroyGenerator.restype = ctypes.c_int
    lib.curandSetPseudoRandomGeneratorSeed.argtypes = [ctypes.c_void_p, ull]
    lib.curandSetPseudoRandomGeneratorSeed.restype = ctypes.c_int
    lib.curandSetGeneratorOffset.argtypes = [ctypes.c_void_p, ull]
    lib.curandSetGeneratorOffset.restype = ctypes.c_int
    lib.curandSetGeneratorOrdering.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.curandSetGeneratorOrdering.restype = ctypes.c_int
    lib.curandSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.curandSetStream.restype = ctypes.c_int
    lib.curandSetQuasiRandomGeneratorDimensions.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.curandSetQuasiRandomGeneratorDimensions.restype = ctypes.c_int
    lib.curandGenerateSeeds.argtypes = [ctypes.c_void_p]
    lib.curandGenerateSeeds.restype = ctypes.c_int
    lib.curandGenerate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, size_t]
    lib.curandGenerate.restype = ctypes.c_int
    lib.curandGenerateLongLong.argtypes = [ctypes.c_void_p, ctypes.c_void_p, size_t]
    lib.curandGenerateLongLong.restype = ctypes.c_int
    lib.curandGenerateUniform.argtypes = [ctypes.c_void_p, ctypes.c_void_p, size_t]
    lib.curandGenerateUniform.restype = ctypes.c_int
    lib.curandGenerateUniformDouble.argtypes = [ctypes.c_void_p, ctypes.c_void_p, size_t]
    lib.curandGenerateUniformDouble.restype = ctypes.c_int
    lib.curandGenerateNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.curandGenerateNormal.restype = ctypes.c_int
    lib.curandGenerateNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.curandGenerateNormalDouble.restype = ctypes.c_int
    lib.curandGenerateLogNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.curandGenerateLogNormal.restype = ctypes.c_int
    lib.curandGenerateLogNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.curandGenerateLogNormalDouble.restype = ctypes.c_int
    lib.curandGeneratePoisson.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        size_t,
        ctypes.c_double,
    ]
    lib.curandGeneratePoisson.restype = ctypes.c_int
    lib.curandGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.curandGetVersion.restype = ctypes.c_int


_bind()


def curand_version() -> int:
    version = ctypes.c_int()
    _check(_libcurand.curandGetVersion(ctypes.byref(version)), "curandGetVersion")
    return int(version.value)


def symbol_matrix() -> dict[str, bool]:
    names = [
        "curandCreateGenerator",
        "curandDestroyGenerator",
        "curandSetPseudoRandomGeneratorSeed",
        "curandSetGeneratorOffset",
        "curandSetGeneratorOrdering",
        "curandSetStream",
        "curandSetQuasiRandomGeneratorDimensions",
        "curandGenerateSeeds",
        "curandGenerate",
        "curandGenerateLongLong",
        "curandGenerateUniform",
        "curandGenerateUniformDouble",
        "curandGenerateNormal",
        "curandGenerateNormalDouble",
        "curandGenerateLogNormal",
        "curandGenerateLogNormalDouble",
        "curandGeneratePoisson",
        "curandGetVersion",
    ]
    return {name: hasattr(_libcurand, name) for name in names}


def library_load_report() -> dict[str, object]:
    return {
        "library": CURAND_LIBRARY,
        "version": curand_version(),
        "platform": sys.platform,
        "symbols": symbol_matrix(),
    }
