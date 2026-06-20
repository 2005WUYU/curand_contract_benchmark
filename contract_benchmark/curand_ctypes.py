from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import torch


CURAND_STATUS_SUCCESS = 0

CURAND_RNG_PSEUDO_XORWOW = 101
CURAND_RNG_PSEUDO_MRG32K3A = 121
CURAND_RNG_PSEUDO_MTGP32 = 141
CURAND_RNG_PSEUDO_MT19937 = 142
CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161
CURAND_RNG_QUASI_SOBOL32 = 201
CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
CURAND_RNG_QUASI_SOBOL64 = 203
CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204

CURAND_ORDERING_PSEUDO_BEST = 100
CURAND_ORDERING_PSEUDO_DEFAULT = 101
CURAND_ORDERING_PSEUDO_SEEDED = 102
CURAND_ORDERING_PSEUDO_LEGACY = 103
CURAND_ORDERING_PSEUDO_DYNAMIC = 104

GENERATOR_TYPES = {
    "xorwow": CURAND_RNG_PSEUDO_XORWOW,
    "mrg32k3a": CURAND_RNG_PSEUDO_MRG32K3A,
    "mtgp32": CURAND_RNG_PSEUDO_MTGP32,
    "mt19937": CURAND_RNG_PSEUDO_MT19937,
    "philox4x32_10": CURAND_RNG_PSEUDO_PHILOX4_32_10,
    "sobol32": CURAND_RNG_QUASI_SOBOL32,
    "scrambled_sobol32": CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
    "sobol64": CURAND_RNG_QUASI_SOBOL64,
    "scrambled_sobol64": CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
}

ORDERINGS = {
    "best": CURAND_ORDERING_PSEUDO_BEST,
    "default": CURAND_ORDERING_PSEUDO_DEFAULT,
    "seeded": CURAND_ORDERING_PSEUDO_SEEDED,
    "legacy": CURAND_ORDERING_PSEUDO_LEGACY,
    "dynamic": CURAND_ORDERING_PSEUDO_DYNAMIC,
}


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


def _stream_ptr(stream: torch.cuda.Stream | None = None) -> ctypes.c_void_p:
    if stream is None:
        stream = torch.cuda.current_stream()
    return ctypes.c_void_p(int(stream.cuda_stream))


class CurandGenerator:
    def __init__(
        self,
        generator: str,
        *,
        seed: int = 12345,
        offset: int = 0,
        ordering: str | None = "legacy",
        dimensions: int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if generator not in GENERATOR_TYPES:
            raise ValueError(f"Unsupported cuRAND generator: {generator}")
        self.generator = generator
        self.generator_type = GENERATOR_TYPES[generator]
        self.seed = int(seed)
        self.offset = int(offset)
        self.ordering = ordering
        self.dimensions = dimensions
        self._handle = ctypes.c_void_p()
        _check(
            _libcurand.curandCreateGenerator(ctypes.byref(self._handle), self.generator_type),
            "curandCreateGenerator",
        )
        try:
            self.set_stream(stream)
            if not self.is_quasi:
                self.set_seed(self.seed)
                if ordering is not None:
                    self.set_ordering(ordering)
            if dimensions is not None:
                self.set_dimensions(dimensions)
            if offset:
                self.set_offset(offset)
        except BaseException:
            self.destroy()
            raise

    @property
    def is_quasi(self) -> bool:
        return self.generator_type >= 200

    @property
    def handle(self) -> ctypes.c_void_p:
        if not self._handle:
            raise RuntimeError("cuRAND generator already destroyed")
        return self._handle

    def __enter__(self) -> "CurandGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self._handle:
            _check(_libcurand.curandDestroyGenerator(self._handle), "curandDestroyGenerator")
            self._handle = ctypes.c_void_p()

    def set_stream(self, stream: torch.cuda.Stream | None = None) -> None:
        _check(_libcurand.curandSetStream(self.handle, _stream_ptr(stream)), "curandSetStream")

    def set_seed(self, seed: int) -> None:
        _check(
            _libcurand.curandSetPseudoRandomGeneratorSeed(self.handle, ctypes.c_ulonglong(seed)),
            "curandSetPseudoRandomGeneratorSeed",
        )
        self.seed = int(seed)

    def set_offset(self, offset: int) -> None:
        _check(
            _libcurand.curandSetGeneratorOffset(self.handle, ctypes.c_ulonglong(offset)),
            "curandSetGeneratorOffset",
        )
        self.offset = int(offset)

    def set_ordering(self, ordering: str) -> None:
        if ordering not in ORDERINGS:
            raise ValueError(f"Unsupported cuRAND ordering: {ordering}")
        _check(
            _libcurand.curandSetGeneratorOrdering(self.handle, ORDERINGS[ordering]),
            "curandSetGeneratorOrdering",
        )
        self.ordering = ordering

    def set_dimensions(self, dimensions: int) -> None:
        _check(
            _libcurand.curandSetQuasiRandomGeneratorDimensions(self.handle, int(dimensions)),
            "curandSetQuasiRandomGeneratorDimensions",
        )
        self.dimensions = int(dimensions)

    def generate_seeds(self) -> None:
        _check(_libcurand.curandGenerateSeeds(self.handle), "curandGenerateSeeds")

    def generate_raw_u32(self, out: torch.Tensor) -> torch.Tensor:
        _check_tensor(out, torch.int32, "curandGenerate")
        _check(_libcurand.curandGenerate(self.handle, ctypes.c_void_p(out.data_ptr()), out.numel()), "curandGenerate")
        return out

    def generate_raw_u64(self, out: torch.Tensor) -> torch.Tensor:
        _check_tensor(out, torch.int64, "curandGenerateLongLong")
        _check(
            _libcurand.curandGenerateLongLong(self.handle, ctypes.c_void_p(out.data_ptr()), out.numel()),
            "curandGenerateLongLong",
        )
        return out

    def generate_uniform_f32(self, out: torch.Tensor) -> torch.Tensor:
        _check_tensor(out, torch.float32, "curandGenerateUniform")
        _check(
            _libcurand.curandGenerateUniform(self.handle, ctypes.c_void_p(out.data_ptr()), out.numel()),
            "curandGenerateUniform",
        )
        return out

    def generate_uniform_f64(self, out: torch.Tensor) -> torch.Tensor:
        _check_tensor(out, torch.float64, "curandGenerateUniformDouble")
        _check(
            _libcurand.curandGenerateUniformDouble(self.handle, ctypes.c_void_p(out.data_ptr()), out.numel()),
            "curandGenerateUniformDouble",
        )
        return out

    def generate_normal_f32(self, out: torch.Tensor, *, mean: float = 0.0, stddev: float = 1.0) -> torch.Tensor:
        _check_tensor(out, torch.float32, "curandGenerateNormal")
        _check(
            _libcurand.curandGenerateNormal(
                self.handle,
                ctypes.c_void_p(out.data_ptr()),
                out.numel(),
                ctypes.c_float(mean),
                ctypes.c_float(stddev),
            ),
            "curandGenerateNormal",
        )
        return out

    def generate_normal_f64(self, out: torch.Tensor, *, mean: float = 0.0, stddev: float = 1.0) -> torch.Tensor:
        _check_tensor(out, torch.float64, "curandGenerateNormalDouble")
        _check(
            _libcurand.curandGenerateNormalDouble(
                self.handle,
                ctypes.c_void_p(out.data_ptr()),
                out.numel(),
                ctypes.c_double(mean),
                ctypes.c_double(stddev),
            ),
            "curandGenerateNormalDouble",
        )
        return out

    def generate_lognormal_f32(self, out: torch.Tensor, *, mean: float = 0.0, stddev: float = 1.0) -> torch.Tensor:
        _check_tensor(out, torch.float32, "curandGenerateLogNormal")
        _check(
            _libcurand.curandGenerateLogNormal(
                self.handle,
                ctypes.c_void_p(out.data_ptr()),
                out.numel(),
                ctypes.c_float(mean),
                ctypes.c_float(stddev),
            ),
            "curandGenerateLogNormal",
        )
        return out

    def generate_lognormal_f64(self, out: torch.Tensor, *, mean: float = 0.0, stddev: float = 1.0) -> torch.Tensor:
        _check_tensor(out, torch.float64, "curandGenerateLogNormalDouble")
        _check(
            _libcurand.curandGenerateLogNormalDouble(
                self.handle,
                ctypes.c_void_p(out.data_ptr()),
                out.numel(),
                ctypes.c_double(mean),
                ctypes.c_double(stddev),
            ),
            "curandGenerateLogNormalDouble",
        )
        return out

    def generate_poisson_u32(self, out: torch.Tensor, *, lambda_val: float) -> torch.Tensor:
        _check_tensor(out, torch.int32, "curandGeneratePoisson")
        _check(
            _libcurand.curandGeneratePoisson(
                self.handle,
                ctypes.c_void_p(out.data_ptr()),
                out.numel(),
                ctypes.c_double(lambda_val),
            ),
            "curandGeneratePoisson",
        )
        return out


def _check_tensor(out: torch.Tensor, dtype: torch.dtype, operation: str) -> None:
    if out.device.type != "cuda":
        raise TypeError(f"{operation}: output tensor must be CUDA, got {out.device}")
    if out.dtype is not dtype:
        raise TypeError(f"{operation}: output dtype must be {dtype}, got {out.dtype}")
    if not out.is_contiguous():
        raise TypeError(f"{operation}: output tensor must be contiguous")


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

