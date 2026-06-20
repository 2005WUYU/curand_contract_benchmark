from __future__ import annotations

import ctypes

import torch

from contract_benchmark.curand_constants import GENERATOR_TYPES, ORDERINGS
from contract_benchmark.curand_library import _check, _libcurand


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
