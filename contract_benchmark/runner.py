from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from contract_benchmark import kernels
from contract_benchmark.adapters import (
    GENERATOR_INFOS,
    capability_matrix,
    curand_generate_by_distribution,
    flagrand_generate_by_distribution,
    flagrand_generate_raw,
    make_curand_generator,
    make_flagrand_generator,
)
from contract_benchmark.curand_ctypes import CurandError, library_load_report
from contract_benchmark.spec import TaskSpec
from contract_benchmark.timing import (
    audit_flags,
    collect_cuda_event_and_wall_us,
    collect_wall_only_us,
    formal_result_from_flags,
)
from contract_benchmark.validation import (
    tensors_equal,
    unsupported,
    validate_finite_output,
    validate_lognormal,
    validate_mask,
    validate_normal,
    validate_poisson,
    validate_raw_tensor,
    validate_uniform,
    validation_error,
    validation_pass,
)


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    sizes: list[int]
    gate_n: int
    warmup: int
    repeats: int
    raw_generators: list[str]
    dist_generators: list[str]
    qrng_generators: list[str]
    poisson_lambdas: list[float]
    fused_ps: list[float]
    many_small_calls: int
    many_small_chunk_n: int


PROFILES = {
    "local_smoke": BenchmarkProfile(
        "local_smoke",
        sizes=[1024, 16384, 65536],
        gate_n=4096,
        warmup=1,
        repeats=3,
        raw_generators=["philox4x32_10", "xorwow", "mrg32k3a"],
        dist_generators=["philox4x32_10"],
        qrng_generators=["sobol32", "sobol64"],
        poisson_lambdas=[1.0, 10.0],
        fused_ps=[0.5],
        many_small_calls=8,
        many_small_chunk_n=1024,
    ),
    "local": BenchmarkProfile(
        "local",
        sizes=[1024, 4096, 65536, 1048576],
        gate_n=16384,
        warmup=3,
        repeats=10,
        raw_generators=["philox4x32_10", "xorwow", "mrg32k3a", "sobol32", "sobol64"],
        dist_generators=["philox4x32_10", "xorwow", "mrg32k3a"],
        qrng_generators=["sobol32", "scrambled_sobol32", "sobol64", "scrambled_sobol64"],
        poisson_lambdas=[0.1, 1.0, 10.0, 64.0],
        fused_ps=[0.1, 0.5, 0.9],
        many_small_calls=64,
        many_small_chunk_n=1024,
    ),
    "h20": BenchmarkProfile(
        "h20",
        sizes=[4096, 16384, 65536, 262144, 1048576, 4194304, 8388608],
        gate_n=65536,
        warmup=5,
        repeats=20,
        raw_generators=["philox4x32_10", "xorwow", "mrg32k3a", "mtgp32", "mt19937", "sobol32", "sobol64"],
        dist_generators=["philox4x32_10", "xorwow", "mrg32k3a"],
        qrng_generators=["sobol32", "scrambled_sobol32", "sobol64", "scrambled_sobol64"],
        poisson_lambdas=[0.1, 1.0, 4.0, 10.0, 32.0, 64.0, 256.0, 1024.0, 10000.0],
        fused_ps=[0.01, 0.1, 0.5, 0.9, 0.99],
        many_small_calls=128,
        many_small_chunk_n=1024,
    ),
}


@dataclass
class BenchmarkContext:
    repo_root: Path
    benchmark_root: Path
    profile: BenchmarkProfile
    specs: dict[str, TaskSpec]
    seed: int = 12345
    offset: int = 0
    device: torch.device = torch.device("cuda")


def collect_environment(profile_name: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    curand_report: dict[str, Any]
    try:
        curand_report = library_load_report()
    except Exception as exc:
        curand_report = {"available": False, "error": str(exc)}
    env = {
        "profile": profile_name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_from_torch": torch.version.cuda,
        "curand": curand_report,
        "time_unix": time.time(),
    }
    if cuda_available:
        env.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    try:
        import triton

        env["triton_version"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        env["triton_version_error"] = str(exc)
    env["git"] = _git_info()
    return env


def run_specs(ctx: BenchmarkContext, selected_specs: list[TaskSpec]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cap_matrix: dict[str, Any] | None = None
    for spec in selected_specs:
        try:
            if spec.task_id == "C0_CAPABILITY_MATRIX":
                cap_matrix = capability_matrix()
                records.extend(_metadata_rows(ctx, spec, cap_matrix))
            elif spec.task_id == "C1_VERSION_SYMBOL_SELFTEST":
                records.append(_metadata_record(ctx, spec, "version_symbol_selftest", collect_environment(ctx.profile.name)))
            elif spec.task_id == "G0_BASIC_CONTRACT":
                records.extend(_run_g0(ctx, spec))
            elif spec.task_id == "G1_DISTRIBUTION_ROUGH_CHECK":
                records.extend(_run_g1(ctx, spec))
            elif spec.task_id == "G2_REPRODUCIBILITY":
                records.extend(_run_g2(ctx, spec))
            elif spec.task_id == "G3_SEQUENCE_COUNTER_BUDGET":
                records.extend(_run_g3(ctx, spec))
            elif spec.task_id == "H0_RAW32_BULK":
                records.extend(_run_bulk_raw32(ctx, spec))
            elif spec.task_id == "H1_RAW64_SOBOL_BULK":
                records.extend(_run_bulk_raw64(ctx, spec))
            elif spec.task_id == "H2_UNIFORM_F32_BULK":
                records.extend(_run_bulk_distribution(ctx, spec, "uniform_f32"))
            elif spec.task_id == "H3_NORMAL_F32_BULK":
                records.extend(_run_bulk_distribution(ctx, spec, "normal_f32"))
            elif spec.task_id == "H4_LOGNORMAL_F32_BULK":
                records.extend(_run_bulk_distribution(ctx, spec, "lognormal_f32"))
            elif spec.task_id == "H5_POISSON_LAMBDA_SWEEP":
                records.extend(_run_poisson(ctx, spec))
            elif spec.task_id == "H6_ORDERING_SWEEP":
                records.extend(_run_ordering_sweep(ctx, spec))
            elif spec.task_id == "I1_GENERATOR_LIFECYCLE":
                records.extend(_run_lifecycle(ctx, spec))
            elif spec.task_id == "I2_CURAND_GENERATE_SEEDS":
                records.extend(_run_generate_seeds(ctx, spec))
            elif spec.task_id == "I3_FIRST_VS_STEADY":
                records.extend(_run_first_vs_steady(ctx, spec))
            elif spec.task_id == "A0_SINGLE_CALL_CURVE":
                records.extend(_run_single_call_curve(ctx, spec))
            elif spec.task_id == "A1_FIXED_TOTAL_MANY_SMALL":
                records.extend(_run_many_small(ctx, spec, calls=ctx.profile.many_small_calls))
            elif spec.task_id == "A2_FIXED_CHUNK_CALLS_SWEEP":
                for calls in [1, max(2, ctx.profile.many_small_calls // 4), ctx.profile.many_small_calls]:
                    records.extend(_run_many_small(ctx, spec, calls=calls))
            elif spec.task_id == "K0_DEVICE_RAW_OUTPUT":
                records.extend(_run_device_raw_output(ctx, spec))
            elif spec.task_id == "K1_DEVICE_UNIFORM_OUTPUT":
                records.extend(_run_device_uniform_output(ctx, spec))
            elif spec.task_id == "M3_DEVICE_DX_FUSED_CONSUME":
                records.extend(_run_m3_device_fused_consume(ctx, spec))
            elif spec.task_id == "E1_COMPILE_SUPPORT_MATRIX":
                records.extend(_run_e1_compile_support_matrix(ctx, spec))
            elif spec.task_id == "F0_THRESHOLD_BERNOULLI":
                records.extend(_run_threshold(ctx, spec))
            elif spec.task_id == "F1_ADD_UNIFORM":
                records.extend(_run_add_uniform(ctx, spec))
            elif spec.task_id == "F2_DROPOUT":
                records.extend(_run_dropout(ctx, spec))
            elif spec.task_id == "M0_PURE_WRITE":
                records.extend(_run_pure_write(ctx, spec))
            elif spec.task_id == "M1_PREGENERATED_CONSUME":
                records.extend(_run_pregenerated_consume(ctx, spec))
            elif spec.task_id == "M2_HOST_BULK_CONSUME":
                records.extend(_run_add_uniform(ctx, spec, task_override="M2_HOST_BULK_CONSUME"))
            elif spec.task_id == "Q0_RAW_SOBOL":
                records.extend(_run_q0(ctx, spec))
            elif spec.task_id == "Q1_SOBOL_D_DIM_UNIT_CUBE":
                records.extend(_run_q1(ctx, spec))
            elif spec.task_id == "E0_HOST_STATUS_MATRIX":
                records.extend(_run_e0(ctx, spec))
            else:
                records.append(_unsupported_record(ctx, spec, "runner", f"No runner implemented for {spec.task_id}"))
        except BaseException as exc:
            records.append(_error_record(ctx, spec, "runner", exc))
    finalize_records(records)
    if cap_matrix is None:
        cap_matrix = capability_matrix()
    return records, cap_matrix


def _run_g0(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.raw_generators + ctx.profile.qrng_generators:
        if generator not in GENERATOR_INFOS:
            continue
        info = GENERATOR_INFOS[generator]
        n = _adjust_n(ctx.profile.gate_n, generator, "raw32" if info.supports_raw32 else "raw64")
        distribution = "raw64" if info.supports_raw64 else "raw32"
        dtype = torch.int64 if info.supports_raw64 else torch.int32
        if info.supports_curand_host:
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            try:
                dims = 1 if info.kind == "qrng" else None
                with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, dimensions=dims) as gen:
                    curand_generate_by_distribution(gen, out, distribution)
                    torch.cuda.synchronize()
                validation = validate_raw_tensor(out, dtype=dtype, n=n)
                records.append(_gate_record(ctx, spec, "curand_host", generator, distribution, n, validation))
            except BaseException as exc:
                records.append(_error_record(ctx, spec, "curand_host", exc, generator=generator, distribution=distribution, n=n))
        if info.supports_flagrand:
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            try:
                gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset, dimensions=1)
                flagrand_generate_raw(out, gen)
                torch.cuda.synchronize()
                validation = validate_raw_tensor(out, dtype=dtype, n=n)
                records.append(_gate_record(ctx, spec, "flagrand_public", generator, distribution, n, validation))
            except BaseException as exc:
                records.append(_error_record(ctx, spec, "flagrand_public", exc, generator=generator, distribution=distribution, n=n))
    return records


def _run_g1(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    n = _adjust_n(ctx.profile.gate_n, generator, "normal_f32")
    for distribution in ("uniform_f32", "normal_f32", "lognormal_f32", "poisson_u32"):
        dtype = _dtype_for_distribution(distribution)
        for backend in ("curand_host", "flagrand_public"):
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            try:
                if backend == "curand_host":
                    with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                        curand_generate_by_distribution(gen, out, distribution, lambda_val=10.0)
                else:
                    gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
                    flagrand_generate_by_distribution(gen, out, distribution, lambda_val=10.0)
                torch.cuda.synchronize()
                validation = _validate_distribution(out, distribution, n, lambda_val=10.0)
                records.append(_gate_record(ctx, spec, backend, generator, distribution, n, validation))
            except BaseException as exc:
                records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution=distribution, n=n))
    return records


def _run_g2(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generators = [
        generator
        for generator in ctx.profile.raw_generators
        if GENERATOR_INFOS[generator].kind == "prng" and GENERATOR_INFOS[generator].supports_raw32
    ]
    for generator in generators:
        info = GENERATOR_INFOS[generator]
        n = _adjust_n(ctx.profile.gate_n, generator, "raw32")
        for backend in ("curand_host", "flagrand_public"):
            if backend == "curand_host" and not info.supports_curand_host:
                continue
            if backend == "flagrand_public" and not info.supports_flagrand:
                continue
            try:
                a = torch.empty(n, device=ctx.device, dtype=torch.int32)
                b = torch.empty_like(a)
                c = torch.empty_like(a)
                e = torch.empty_like(a)
                d = torch.empty_like(a) if info.supports_offset else None
                if backend == "curand_host":
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(a)
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(b)
                    with make_curand_generator(generator, seed=ctx.seed + 1, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(c)
                    if d is not None:
                        with make_curand_generator(generator, seed=ctx.seed, offset=n, ordering="legacy") as gen:
                            gen.generate_raw_u32(d)
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(e)
                        gen.generate_raw_u32(e)
                else:
                    gen_a = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    gen_b = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    gen_c = make_flagrand_generator(generator, seed=ctx.seed + 1, offset=0)
                    gen_e = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    flagrand_generate_raw(a, gen_a)
                    flagrand_generate_raw(b, gen_b)
                    flagrand_generate_raw(c, gen_c)
                    if d is not None:
                        gen_d = make_flagrand_generator(generator, seed=ctx.seed, offset=n)
                        flagrand_generate_raw(d, gen_d)
                    flagrand_generate_raw(e, gen_e)
                    flagrand_generate_raw(e, gen_e)
                torch.cuda.synchronize()
                checks: dict[str, Any] = {
                    "same_seed_same_output": tensors_equal(a, b),
                    "changed_seed_changes_output": not tensors_equal(a, c),
                    "same_generator_second_call_advances": not tensors_equal(a, e),
                    "offset_check_applicable": "yes" if info.supports_offset else "not_supported_by_generator",
                }
                if d is not None:
                    checks["changed_offset_changes_output"] = not tensors_equal(a, d)
                validation = validation_pass(checks)
                records.append(_gate_record(ctx, spec, backend, generator, "raw32", n, validation))
            except BaseException as exc:
                records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution="raw32", n=n))
    return records


def _run_g3(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records = []
    for n in ctx.profile.sizes:
        n = _adjust_n(n, "philox4x32_10", "uniform_f32")
        counters = (n + 3) // 4
        checks = {
            "n_multiple_of_4": n % 4 == 0,
            "counter_count_positive": counters > 0,
            "counter_count": counters,
            "reserved_counter_increment": counters,
            "max_lane_index": n - 1,
        }
        records.append(_gate_record(ctx, spec, "flagrand_fused_philox", "philox4x32_10", "philox_counter_budget", n, validation_pass(checks)))
    return records


def _run_bulk_raw32(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.raw_generators:
        info = GENERATOR_INFOS.get(generator)
        if info is None or not info.supports_raw32:
            continue
        for n0 in ctx.profile.sizes:
            n = _adjust_n(n0, generator, "raw32")
            records.extend(_measure_curand_and_flagrand_raw(ctx, spec, generator, n, "raw32", torch.int32))
    return records


def _run_bulk_raw64(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ("sobol64", "scrambled_sobol64"):
        if generator not in ctx.profile.qrng_generators and generator != "sobol64":
            continue
        for n in ctx.profile.sizes:
            records.extend(_measure_curand_and_flagrand_raw(ctx, spec, generator, n, "raw64", torch.int64))
    return records


def _run_bulk_distribution(ctx: BenchmarkContext, spec: TaskSpec, distribution: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.dist_generators:
        for n0 in ctx.profile.sizes:
            n = _adjust_n(n0, generator, distribution)
            records.extend(_measure_curand_and_flagrand_distribution(ctx, spec, generator, n, distribution))
    return records


def _run_poisson(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    for lambda_val in ctx.profile.poisson_lambdas:
        for n0 in ctx.profile.sizes:
            n = _adjust_n(n0, generator, "poisson_u32")
            records.extend(
                _measure_curand_and_flagrand_distribution(
                    ctx,
                    spec,
                    generator,
                    n,
                    "poisson_u32",
                    parameters={"lambda": lambda_val},
                    lambda_val=lambda_val,
                )
            )
    return records


def _run_ordering_sweep(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    n = _adjust_n(max(ctx.profile.sizes), generator, "raw32")
    for ordering in ("legacy", "default", "best", "dynamic", "seeded"):
        out = torch.empty(n, device=ctx.device, dtype=torch.int32)
        try:
            with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering=ordering) as gen:
                validation = _validate_after(lambda: gen.generate_raw_u32(out), lambda: validate_raw_tensor(out, dtype=torch.int32, n=n))
                timing = collect_cuda_event_and_wall_us(lambda: gen.generate_raw_u32(out), warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(
                    ctx,
                    spec,
                    "curand_host_ordering",
                    "curand_host_api",
                    generator,
                    "raw32",
                    n,
                    timing,
                    validation,
                    parameters={"ordering": ordering},
                    ordering=ordering,
                    comparison_key=f"{spec.task_id}:{generator}:{n}",
                    is_baseline=ordering == "legacy",
                    baseline_id="curand_host_legacy",
                )
            )
        except CurandError as exc:
            records.append(
                _unsupported_record(
                    ctx,
                    spec,
                    "curand_host_ordering",
                    f"ordering={ordering} unsupported for {generator}: cuRAND status {exc.status}",
                    generator=generator,
                    distribution="raw32",
                    n=n,
                    parameters={"ordering": ordering},
                    comparison_key=f"{spec.task_id}:{generator}:{n}",
                    baseline_id="curand_host_legacy",
                )
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, "curand_host_ordering", exc, generator=generator, distribution="raw32", n=n, parameters={"ordering": ordering}))
    return records


def _run_lifecycle(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"

    def curand_once() -> None:
        gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")
        gen.destroy()

    def flagrand_once() -> object:
        return make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)

    for backend, api_surface, run_once in (
        ("curand_host_lifecycle", "curand_host_api", curand_once),
        ("flagrand_lifecycle", "flagrand_public_api", flagrand_once),
    ):
        try:
            validation = validation_pass({"status_success": True})
            timing = collect_wall_only_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats, sync_cuda=True)
            records.append(
                _timed_record(
                    ctx,
                    spec,
                    backend,
                    api_surface,
                    generator,
                    "lifecycle",
                    0,
                    timing,
                    validation,
                    comparison_key=f"{spec.task_id}:{generator}",
                    is_baseline=backend.startswith("curand"),
                    baseline_id="curand_host_lifecycle",
                )
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution="lifecycle", n=0))
    return records


def _run_generate_seeds(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in [g for g in ctx.profile.raw_generators if GENERATOR_INFOS[g].kind == "prng"]:
        try:
            with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                validation = _validate_after(gen.generate_seeds, lambda: validation_pass({"curand_status_success": True}))
                timing = collect_cuda_event_and_wall_us(gen.generate_seeds, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(
                    ctx,
                    spec,
                    "curand_host_generate_seeds",
                    "curand_host_api",
                    generator,
                    "generate_seeds",
                    0,
                    timing,
                    validation,
                    comparison_key=f"{spec.task_id}:{generator}",
                    is_baseline=True,
                    baseline_id="curand_host_generate_seeds",
                )
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, "curand_host_generate_seeds", exc, generator=generator, distribution="generate_seeds", n=0))
    records.append(_unsupported_record(ctx, spec, "flagrand_no_exact_generate_seeds", "FlagRand has no Host API equivalent to curandGenerateSeeds."))
    return records


def _make_uniform_once(
    ctx: BenchmarkContext,
    backend: str,
    generator: str,
    out: torch.Tensor,
) -> tuple[Callable[[], object], Callable[[], None]]:
    if backend == "curand_host":
        gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")

        def cleanup() -> None:
            gen.destroy()

        return lambda: gen.generate_uniform_f32(out), cleanup

    gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
    return lambda: flagrand_generate_by_distribution(gen, out, "uniform_f32"), lambda: None


def _run_first_vs_steady(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    n = _adjust_n(ctx.profile.gate_n, generator, "uniform_f32")
    for backend in ("curand_host", "flagrand_public"):
        try:
            if backend == "curand_host":
                api_surface = "curand_host_api"
            else:
                api_surface = "flagrand_public_api"

            validation_out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            validation_run, validation_cleanup = _make_uniform_once(ctx, backend, generator, validation_out)
            try:
                validation = _validate_after(validation_run, lambda: validate_uniform(validation_out, n=n))
            finally:
                validation_cleanup()

            timings = []
            for phase, warmup_iters, repeats in (
                ("first", 0, 1),
                ("steady", ctx.profile.warmup, ctx.profile.repeats),
            ):
                out = torch.empty(n, device=ctx.device, dtype=torch.float32)
                run_once, cleanup = _make_uniform_once(ctx, backend, generator, out)
                try:
                    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=warmup_iters, repeats=repeats)
                finally:
                    cleanup()
                timings.append((phase, timing))

            for phase, timing in timings:
                records.append(
                    _timed_record(
                        ctx,
                        spec,
                        f"{backend}_{phase}",
                        api_surface,
                        generator,
                        "uniform_f32",
                        n,
                        timing,
                        validation,
                        parameters={
                            "phase": phase,
                            "timing_semantics": "fresh_generator_after_separate_validation",
                            "first_warmup_iters": 0,
                        },
                        comparison_key=f"{spec.task_id}:{phase}:{generator}:{n}",
                        is_baseline=backend == "curand_host",
                        baseline_id=f"curand_host_{phase}",
                    )
                )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution="uniform_f32", n=n))
    return records


def _run_single_call_curve(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    return _run_bulk_distribution(ctx, spec, "uniform_f32")


def _run_many_small(ctx: BenchmarkContext, spec: TaskSpec, *, calls: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    chunk_n = _adjust_n(ctx.profile.many_small_chunk_n, generator, "uniform_f32")
    total_n = chunk_n * calls
    for backend in ("curand_host", "flagrand_public"):
        chunks = [torch.empty(chunk_n, device=ctx.device, dtype=torch.float32) for _ in range(calls)]
        try:
            if backend == "curand_host":
                gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")
                api_surface = "curand_host_api"

                def run_once() -> None:
                    for chunk in chunks:
                        gen.generate_uniform_f32(chunk)

            else:
                gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
                api_surface = "flagrand_public_api"

                def run_once() -> None:
                    for chunk in chunks:
                        flagrand_generate_by_distribution(gen, chunk, "uniform_f32")

            validation = _validate_after(run_once, lambda: validate_uniform(chunks[-1], n=chunk_n))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            if backend == "curand_host":
                gen.destroy()
            records.append(
                _timed_record(
                    ctx,
                    spec,
                    backend,
                    api_surface,
                    generator,
                    "uniform_f32",
                    total_n,
                    timing,
                    validation,
                    parameters={"calls": calls, "chunk_n": chunk_n, "total_n": total_n},
                    comparison_key=f"{spec.task_id}:{calls}:{chunk_n}",
                    is_baseline=backend == "curand_host",
                    baseline_id="curand_host_many_small",
                    output_bytes=total_n * 4,
                    item_count=total_n,
                )
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution="uniform_f32", n=total_n, parameters={"calls": calls, "chunk_n": chunk_n}))
    return records


def _run_threshold(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in ctx.profile.fused_ps:
        for n0 in ctx.profile.sizes:
            n = _adjust_n(n0, "philox4x32_10", "uniform_f32")
            u = torch.empty(n, device=ctx.device, dtype=torch.float32)
            mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
            with make_curand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                run_base = lambda: (gen.generate_uniform_f32(u), kernels.threshold_from_uniform(u, mask, p=p))
                validation = _validate_after(run_base, lambda: validate_mask(mask, n=n, p=p))
                timing = collect_cuda_event_and_wall_us(run_base, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(ctx, spec, "curand_host_uniform_plus_threshold", "curand_host_api+triton_consume", "philox4x32_10", "uniform_threshold", n, timing, validation, parameters={"p": p}, comparison_key=f"{spec.task_id}:{p}:{n}", is_baseline=True, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=n * 4)
            )
            mask2 = torch.empty(n, device=ctx.device, dtype=torch.uint8)
            run_fused = lambda: kernels.fused_philox_threshold(mask2, seed=ctx.seed, p=p)
            validation2 = _validate_after(run_fused, lambda: validate_mask(mask2, n=n, p=p))
            timing2 = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(ctx, spec, "flagrand_fused_philox", "flagrand_benchmark_kernel", "philox4x32_10", "uniform_threshold", n, timing2, validation2, parameters={"p": p}, comparison_key=f"{spec.task_id}:{p}:{n}", is_baseline=False, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=0)
            )
            records.extend(_device_fused_unsupported_rows(ctx, spec, n=n, distribution="uniform_threshold", parameters={"p": p}, comparison_key=f"{spec.task_id}:{p}:{n}"))
    return records


def _run_add_uniform(ctx: BenchmarkContext, spec: TaskSpec, *, task_override: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_id = task_override or spec.task_id
    alpha = 0.25
    for n0 in ctx.profile.sizes:
        n = _adjust_n(n0, "philox4x32_10", "uniform_f32")
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        u = torch.empty(n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        with make_curand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
            run_base = lambda: (gen.generate_uniform_f32(u), kernels.consume_add_uniform(x, u, out, alpha=alpha))
            validation = _validate_after(run_base, lambda: validate_finite_output(out, n=n))
            timing = collect_cuda_event_and_wall_us(run_base, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(
            _timed_record(ctx, spec, "curand_host_bulk_plus_consume", "curand_host_api+triton_consume", "philox4x32_10", "uniform_add_consume", n, timing, validation, parameters={"alpha": alpha}, comparison_key=f"{task_id}:{n}", is_baseline=True, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=n * 4)
        )
        out2 = torch.empty_like(out)
        run_fused = lambda: kernels.fused_philox_add_uniform(x, out2, seed=ctx.seed, alpha=alpha)
        validation2 = _validate_after(run_fused, lambda: validate_finite_output(out2, n=n))
        timing2 = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        rec = _timed_record(ctx, spec, "flagrand_fused_philox", "flagrand_benchmark_kernel", "philox4x32_10", "uniform_add_consume", n, timing2, validation2, parameters={"alpha": alpha}, comparison_key=f"{task_id}:{n}", is_baseline=False, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=0)
        if task_override:
            rec["task_id"] = task_override
        records.append(rec)
        if task_override is None:
            records.extend(_device_fused_unsupported_rows(ctx, spec, n=n, distribution="uniform_add_consume", parameters={"alpha": alpha}, comparison_key=f"{task_id}:{n}"))
    return records


def _run_dropout(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in ctx.profile.fused_ps:
        for n0 in ctx.profile.sizes:
            n = _adjust_n(n0, "philox4x32_10", "uniform_f32")
            x = torch.ones(n, device=ctx.device, dtype=torch.float32)
            u = torch.empty(n, device=ctx.device, dtype=torch.float32)
            out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
            with make_curand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                run_base = lambda: (gen.generate_uniform_f32(u), kernels.dropout_from_uniform(x, u, out, mask, p=p))
                validation = _validate_after(run_base, lambda: _merge_validations(validate_finite_output(out, n=n), validate_mask(mask, n=n, p=p)))
                timing = collect_cuda_event_and_wall_us(run_base, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(ctx, spec, "curand_host_uniform_plus_dropout", "curand_host_api+triton_consume", "philox4x32_10", "dropout", n, timing, validation, parameters={"p": p, "rule": "u<=p", "scaling": "inverted"}, comparison_key=f"{spec.task_id}:{p}:{n}", is_baseline=True, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=n * 4)
            )
            out2 = torch.empty_like(out)
            mask2 = torch.empty_like(mask)
            run_fused = lambda: kernels.fused_philox_dropout(x, out2, mask2, seed=ctx.seed, p=p)
            validation2 = _validate_after(run_fused, lambda: _merge_validations(validate_finite_output(out2, n=n), validate_mask(mask2, n=n, p=p)))
            timing2 = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(ctx, spec, "flagrand_fused_philox", "flagrand_benchmark_kernel", "philox4x32_10", "dropout", n, timing2, validation2, parameters={"p": p, "rule": "u<=p", "scaling": "inverted"}, comparison_key=f"{spec.task_id}:{p}:{n}", is_baseline=False, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=0)
            )
            records.extend(_device_fused_unsupported_rows(ctx, spec, n=n, distribution="dropout", parameters={"p": p}, comparison_key=f"{spec.task_id}:{p}:{n}"))
    return records


def _run_pure_write(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records = []
    for n in ctx.profile.sizes:
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_once = lambda: kernels.pure_write_f32(out, value=1.0)
        validation = _validate_after(run_once, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(_timed_record(ctx, spec, "triton_pure_write", "flagrand_benchmark_kernel", "none", "pure_write_f32", n, timing, validation, comparison_key=f"{spec.task_id}:{n}", is_baseline=True, output_bytes=n * 4))
    return records


def _run_pregenerated_consume(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records = []
    alpha = 0.25
    for n in ctx.profile.sizes:
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        u = torch.rand(n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_once = lambda: kernels.consume_add_uniform(x, u, out, alpha=alpha)
        validation = _validate_after(run_once, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(_timed_record(ctx, spec, "triton_pregenerated_consume", "flagrand_benchmark_kernel", "none", "consume_pregenerated_uniform", n, timing, validation, parameters={"alpha": alpha}, comparison_key=f"{spec.task_id}:{n}", is_baseline=True, output_bytes=n * 8))
    return records


def _run_q0(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.qrng_generators:
        info = GENERATOR_INFOS[generator]
        distribution = "raw64" if info.supports_raw64 else "raw32"
        dtype = torch.int64 if info.supports_raw64 else torch.int32
        for n in ctx.profile.sizes:
            records.extend(_measure_curand_and_flagrand_raw(ctx, spec, generator, n, distribution, dtype, dimensions=1))
    return records


def _run_q1(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dimensions = 4
    points = max(1024, ctx.profile.gate_n // dimensions)
    n = points * dimensions
    for generator in ("sobol32", "scrambled_sobol32"):
        for backend in ("curand_host", "flagrand_public"):
            out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            try:
                if backend == "curand_host":
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, dimensions=dimensions) as gen:
                        run_once = lambda: gen.generate_uniform_f32(out)
                        api_surface = "curand_host_api"
                        validation = _validate_after(run_once, lambda: validate_uniform(out, n=n, low_open=True))
                        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
                else:
                    gen = make_flagrand_generator(generator, seed=ctx.seed, offset=0, dimensions=dimensions)
                    run_once = lambda: flagrand_generate_by_distribution(gen, out, "uniform_f32")
                    api_surface = "flagrand_public_api"
                    validation = _validate_after(run_once, lambda: validate_uniform(out, n=n))
                    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
                records.append(
                    _timed_record(ctx, spec, backend, api_surface, generator, "sobol_unit_cube_f32", n, timing, validation, parameters={"points": points, "dimensions": dimensions, "layout": "dimension_major_flattened"}, comparison_key=f"{spec.task_id}:{generator}:{points}:{dimensions}", is_baseline=backend == "curand_host", baseline_id="curand_host_sobol")
                )
            except BaseException as exc:
                records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution="sobol_unit_cube_f32", n=n, parameters={"points": points, "dimensions": dimensions}))
    return records


def _run_e0(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cases = [
        ("curand_invalid_lambda", "curand_host", _e0_curand_invalid_lambda, True),
        ("curand_odd_normal_n", "curand_host", _e0_curand_odd_normal_n, False),
        ("curand_invalid_dimensions", "curand_host", _e0_curand_invalid_dimensions, True),
        ("flagrand_invalid_lambda", "flagrand_public", _e0_flagrand_invalid_lambda, True),
        ("flagrand_odd_philox_n", "flagrand_public", _e0_flagrand_odd_philox_n, True),
    ]
    for case_id, backend, fn, expected_raised in cases:
        try:
            outcome = fn(ctx)
            validation = validation_pass(
                {
                    "outcome_recorded": True,
                    "expected_raised": expected_raised,
                    "raised_matches_expected": bool(outcome.get("raised")) == expected_raised,
                    "failure_not_aggregated_as_speedup": True,
                }
            )
            validation["observed_error"] = outcome
        except BaseException as exc:
            validation = validation_error(exc)
        records.append(
            _base_record(ctx, spec, backend, "robustness", "various", "error_case", 0, validation=validation, parameters={"case": case_id})
        )
    return records


def _run_device_raw_output(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    parameters = _legacy_device_mapping_parameters(ctx)
    if ext is None or not hasattr(ext, "philox_raw_u32"):
        records.append(
            _unsupported_record(
                ctx,
                spec,
                "curand_legacy_device_output",
                ext_reason or "legacy Device API extension does not expose philox_raw_u32",
                generator="philox4x32_10",
                distribution="raw32",
                parameters=parameters,
                baseline_id="curand_legacy_device_output",
            )
        )
    else:
        for n in ctx.profile.sizes:
            out = torch.empty(n, device=ctx.device, dtype=torch.int32)
            run_once = lambda: ext.philox_raw_u32(out, ctx.seed, ctx.offset)
            validation = _validate_after(run_once, lambda: validate_raw_tensor(out, dtype=torch.int32, n=n))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(
                    ctx,
                    spec,
                    "curand_legacy_device_output",
                    "legacy_device_api_extension",
                    "philox4x32_10",
                    "raw32",
                    n,
                    timing,
                    validation,
                    parameters=parameters,
                    comparison_key=f"{spec.task_id}:legacy_device:philox4x32_10:{n}",
                    is_baseline=True,
                    baseline_id="curand_legacy_device_output",
                    temporary_bytes=0,
                    output_bytes=n * 4,
                )
            )
    records.append(_curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="raw32"))
    return records


def _run_device_uniform_output(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    parameters = _legacy_device_mapping_parameters(ctx)
    if ext is None or not hasattr(ext, "philox_uniform"):
        records.append(
            _unsupported_record(
                ctx,
                spec,
                "curand_legacy_device_output",
                ext_reason or "legacy Device API extension does not expose philox_uniform",
                generator="philox4x32_10",
                distribution="uniform_f32",
                parameters=parameters,
                baseline_id="curand_legacy_device_output",
            )
        )
    else:
        for n in ctx.profile.sizes:
            out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            run_once = lambda: ext.philox_uniform(out, ctx.seed, ctx.offset)
            validation = _validate_after(run_once, lambda: validate_uniform(out, n=n, low_open=True))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                _timed_record(
                    ctx,
                    spec,
                    "curand_legacy_device_output",
                    "legacy_device_api_extension",
                    "philox4x32_10",
                    "uniform_f32",
                    n,
                    timing,
                    validation,
                    parameters=parameters,
                    comparison_key=f"{spec.task_id}:legacy_device:philox4x32_10:{n}",
                    is_baseline=True,
                    baseline_id="curand_legacy_device_output",
                    temporary_bytes=0,
                    output_bytes=n * 4,
                )
            )
    records.append(_curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="uniform_f32"))
    return records


def _run_m3_device_fused_consume(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    alpha = 0.25
    if ext is None:
        records.append(
            _unsupported_record(
                ctx,
                spec,
                "curand_legacy_device_fused",
                ext_reason or "legacy Device API extension unavailable",
                generator="philox4x32_10",
                distribution="uniform_add_consume",
                parameters={"alpha": alpha, **_legacy_device_mapping_parameters(ctx)},
                baseline_id="curand_legacy_device_fused",
            )
        )
        records.append(_curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="uniform_add_consume"))
        return records

    for n0 in ctx.profile.sizes:
        n = _adjust_n(n0, "philox4x32_10", "uniform_f32")
        parameters = {"alpha": alpha, "operation": "y=x+alpha*(u-0.5)", **_legacy_device_mapping_parameters(ctx)}
        comparison_key = f"{spec.task_id}:uniform_add_consume:{n}:legacy_device"
        try:
            legacy_rows = _run_legacy_device_fused_extension(
                ctx,
                spec,
                ext,
                n=n,
                distribution="uniform_add_consume",
                parameters=parameters,
                comparison_key=comparison_key,
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, "curand_legacy_device_fused", exc, generator="philox4x32_10", distribution="uniform_add_consume", n=n, parameters=parameters))
            records.append(_curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="uniform_add_consume", n=n, parameters={"alpha": alpha, "operation": "y=x+alpha*(u-0.5)"}))
            continue
        records.extend(legacy_rows)

        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_fused = lambda: kernels.fused_philox_add_uniform(x, out, seed=ctx.seed, alpha=alpha)
        validation = _validate_after(run_fused, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(
            _timed_record(
                ctx,
                spec,
                "flagrand_fused_philox",
                "flagrand_benchmark_kernel",
                "philox4x32_10",
                "uniform_add_consume",
                n,
                timing,
                validation,
                parameters={"alpha": alpha, "operation": "y=x+alpha*(u-0.5)"},
                comparison_key=comparison_key,
                is_baseline=False,
                baseline_id="curand_legacy_device_fused",
                temporary_bytes=0,
                output_bytes=n * 4,
            )
        )
        records.append(
            _curanddx_unsupported_record(
                ctx,
                spec,
                generator="philox4x32_10",
                distribution="uniform_add_consume",
                n=n,
                parameters={"alpha": alpha, "operation": "y=x+alpha*(u-0.5)"},
            )
        )
    return records


def _run_e1_compile_support_matrix(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    required_symbols = [
        "philox_raw_u32",
        "philox_uniform",
        "philox_add_uniform",
        "philox_threshold",
        "philox_dropout",
    ]
    if ext is None:
        records.append(
            _unsupported_record(
                ctx,
                spec,
                "curand_legacy_device",
                ext_reason or "legacy Device API extension unavailable",
                generator="philox4x32_10",
                distribution="compile_support",
                parameters={"required_symbols": required_symbols},
            )
        )
    else:
        checks = {"extension_importable": True}
        checks.update({f"has_{symbol}": hasattr(ext, symbol) for symbol in required_symbols})
        validation = validation_pass(checks)
        record = _base_record(
            ctx,
            spec,
            "curand_legacy_device",
            "compile_support_matrix",
            "philox4x32_10",
            "compile_support",
            0,
            validation=validation,
            parameters={"required_symbols": required_symbols, "source": "native/curand_contract_device_ext.cu"},
        )
        record.update({"formal_result": validation.get("status") == "pass", "audit_flags": []})
        records.append(record)

    records.append(_curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="compile_support"))
    return records


def _legacy_device_mapping_parameters(ctx: BenchmarkContext) -> dict[str, Any]:
    return {
        "device_mapping": "curand_init(seed, sequence=linear_index, offset=absolute_offset)",
        "absolute_offset": ctx.offset,
        "host_order_exact_match": False,
    }


def _curanddx_unsupported_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    *,
    generator: str,
    distribution: str,
    n: int = 0,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cap = capability_matrix()
    return _unsupported_record(
        ctx,
        spec,
        "curanddx",
        cap.get("curanddx", {}).get("unsupported_reason") or "cuRANDDx headers/build integration are not configured in this local repository.",
        generator=generator,
        distribution=distribution,
        n=n,
        parameters=parameters,
        baseline_id="curanddx",
    )


def _measure_curand_and_flagrand_raw(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    generator: str,
    n: int,
    distribution: str,
    dtype: torch.dtype,
    *,
    dimensions: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for backend in ("curand_host", "flagrand_public"):
        out = torch.empty(n, device=ctx.device, dtype=dtype)
        try:
            if backend == "curand_host":
                gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy", dimensions=dimensions)
                run_once = lambda: curand_generate_by_distribution(gen, out, distribution)
                api_surface = "curand_host_api"
            else:
                gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset, dimensions=dimensions)
                run_once = lambda: flagrand_generate_by_distribution(gen, out, distribution)
                api_surface = "flagrand_public_api"
            validation = _validate_after(run_once, lambda: validate_raw_tensor(out, dtype=dtype, n=n))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            if backend == "curand_host":
                gen.destroy()
            records.append(
                _timed_record(ctx, spec, backend, api_surface, generator, distribution, n, timing, validation, comparison_key=f"{spec.task_id}:{generator}:{distribution}:{n}:{dimensions}", is_baseline=backend == "curand_host", baseline_id="curand_host_legacy", output_bytes=n * out.element_size())
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution=distribution, n=n))
    return records


def _measure_curand_and_flagrand_distribution(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    generator: str,
    n: int,
    distribution: str,
    *,
    parameters: dict[str, Any] | None = None,
    lambda_val: float = 10.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dtype = _dtype_for_distribution(distribution)
    params = parameters or {}
    for backend in ("curand_host", "flagrand_public"):
        out = torch.empty(n, device=ctx.device, dtype=dtype)
        try:
            if backend == "curand_host":
                gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")
                run_once = lambda: curand_generate_by_distribution(gen, out, distribution, lambda_val=lambda_val)
                api_surface = "curand_host_api"
            else:
                gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
                run_once = lambda: flagrand_generate_by_distribution(gen, out, distribution, lambda_val=lambda_val)
                api_surface = "flagrand_public_api"
            validation = _validate_after(run_once, lambda: _validate_distribution(out, distribution, n, lambda_val=lambda_val))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            if backend == "curand_host":
                gen.destroy()
            records.append(
                _timed_record(ctx, spec, backend, api_surface, generator, distribution, n, timing, validation, parameters=params, comparison_key=f"{spec.task_id}:{generator}:{distribution}:{n}:{json.dumps(params, sort_keys=True)}", is_baseline=backend == "curand_host", baseline_id="curand_host_legacy", output_bytes=n * out.element_size())
            )
        except BaseException as exc:
            records.append(_error_record(ctx, spec, backend, exc, generator=generator, distribution=distribution, n=n, parameters=params))
    return records


def _device_fused_unsupported_rows(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    *,
    n: int,
    distribution: str,
    parameters: dict[str, Any],
    comparison_key: str,
) -> list[dict[str, Any]]:
    cap = capability_matrix()
    rows = []
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    ext, ext_reason = find_built_curand_device_extension()
    if ext is None:
        rows.append(
            _unsupported_record(
                ctx,
                spec,
                "curand_legacy_device_fused",
                cap.get("device_api_extension", {}).get("unsupported_reason") or ext_reason or "legacy Device API extension unavailable",
                generator="philox4x32_10",
                distribution=distribution,
                n=n,
                parameters=parameters,
                comparison_key=f"{comparison_key}:legacy_device",
                baseline_id="curand_legacy_device_fused",
            )
        )
    else:
        try:
            rows.extend(_run_legacy_device_fused_extension(ctx, spec, ext, n=n, distribution=distribution, parameters=parameters, comparison_key=f"{comparison_key}:legacy_device"))
        except BaseException as exc:
            rows.append(_error_record(ctx, spec, "curand_legacy_device_fused", exc, generator="philox4x32_10", distribution=distribution, n=n, parameters=parameters))
    rows.append(
        _unsupported_record(
            ctx,
            spec,
            "curanddx_fused",
            cap.get("curanddx", {}).get("unsupported_reason") or "cuRANDDx unavailable",
            generator="philox4x32_10",
            distribution=distribution,
            n=n,
            parameters=parameters,
            comparison_key=comparison_key,
            baseline_id="curanddx_fused",
        )
    )
    return rows


def _run_legacy_device_fused_extension(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    ext: Any,
    *,
    n: int,
    distribution: str,
    parameters: dict[str, Any],
    comparison_key: str,
) -> list[dict[str, Any]]:
    p = float(parameters.get("p", 0.5))
    alpha = float(parameters.get("alpha", 0.25))
    if distribution == "uniform_threshold":
        mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
        run_once = lambda: ext.philox_threshold(mask, ctx.seed, ctx.offset, p)
        validation = _validate_after(run_once, lambda: validate_mask(mask, n=n, p=p))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        output_bytes = n
    elif distribution == "uniform_add_consume":
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_once = lambda: ext.philox_add_uniform(x, out, ctx.seed, ctx.offset, alpha)
        validation = _validate_after(run_once, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        output_bytes = n * 4
    elif distribution == "dropout":
        x = torch.ones(n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
        run_once = lambda: ext.philox_dropout(x, out, mask, ctx.seed, ctx.offset, p)
        validation = _validate_after(run_once, lambda: _merge_validations(validate_finite_output(out, n=n), validate_mask(mask, n=n, p=p)))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        output_bytes = n * 5
    else:
        return [_unsupported_record(ctx, spec, "curand_legacy_device_fused", f"extension has no runner for distribution={distribution}", generator="philox4x32_10", distribution=distribution, n=n, parameters=parameters)]
    return [
        _timed_record(
            ctx,
            spec,
            "curand_legacy_device_fused",
            "legacy_device_api_extension",
            "philox4x32_10",
            distribution,
            n,
            timing,
            validation,
            parameters=parameters,
            comparison_key=comparison_key,
            is_baseline=True,
            baseline_id="curand_legacy_device_fused",
            temporary_bytes=0,
            output_bytes=output_bytes,
        )
    ]


def _validate_after(run_once: Callable[[], object], validate: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        run_once()
        torch.cuda.synchronize()
        return validate()
    except BaseException as exc:
        return validation_error(exc)


def _validate_distribution(out: torch.Tensor, distribution: str, n: int, *, lambda_val: float = 10.0) -> dict[str, Any]:
    if distribution == "uniform_f32":
        return validate_uniform(out, n=n, low_open=True)
    if distribution == "normal_f32":
        return validate_normal(out, n=n, mean=0.0, stddev=1.0)
    if distribution == "lognormal_f32":
        return validate_lognormal(out, n=n)
    if distribution == "poisson_u32":
        return validate_poisson(out, n=n, lambda_val=lambda_val)
    if distribution == "raw32":
        return validate_raw_tensor(out, dtype=torch.int32, n=n)
    if distribution == "raw64":
        return validate_raw_tensor(out, dtype=torch.int64, n=n)
    return validate_finite_output(out, n=n)


def _merge_validations(*validations: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    status = "pass"
    for idx, validation in enumerate(validations):
        if validation.get("status") != "pass":
            status = "fail"
        for key, value in validation.get("checks", {}).items():
            checks[f"{idx}_{key}"] = value
    return {"status": status, "checks": checks}


def _dtype_for_distribution(distribution: str) -> torch.dtype:
    if distribution in {"raw32", "poisson_u32"}:
        return torch.int32
    if distribution == "raw64":
        return torch.int64
    if distribution.endswith("_f64"):
        return torch.float64
    return torch.float32


def _adjust_n(n: int, generator: str, distribution: str) -> int:
    value = int(n)
    if generator == "philox4x32_10" or distribution in {"normal_f32", "lognormal_f32"}:
        value = max(4, value + (-value % 4))
    if distribution in {"normal_f32", "lognormal_f32"} and value % 2:
        value += 1
    return value


def _timed_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    backend: str,
    api_surface: str,
    generator: str,
    distribution: str,
    n: int,
    timing,
    validation: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
    comparison_key: str,
    is_baseline: bool,
    baseline_id: str | None = None,
    ordering: str | None = "legacy",
    temporary_bytes: int | str | None = "not_exposed",
    output_bytes: int | None = None,
    item_count: int | None = None,
) -> dict[str, Any]:
    flags = audit_flags(timing)
    record = _base_record(
        ctx,
        spec,
        backend,
        api_surface,
        generator,
        distribution,
        n,
        validation=validation,
        parameters=parameters,
        ordering=ordering,
    )
    record.update(timing.to_record())
    record.update(
        {
            "comparison_key": comparison_key,
            "is_baseline": is_baseline,
            "baseline_id": baseline_id,
            "temporary_bytes": temporary_bytes,
            "audit_flags": flags,
            "formal_result": validation.get("status") == "pass" and formal_result_from_flags(flags),
        }
    )
    median = record.get("median_gpu_us")
    items = item_count if item_count is not None else n
    bytes_out = output_bytes if output_bytes is not None else _estimate_output_bytes(distribution, n)
    if median and median > 0 and items:
        record["items_per_second"] = items / (median * 1e-6)
    if median and median > 0 and bytes_out:
        record["bytes_per_second"] = bytes_out / (median * 1e-6)
    return record


def _base_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    backend: str,
    api_surface: str,
    generator: str,
    distribution: str,
    n: int,
    *,
    validation: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    ordering: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": spec.task_id,
        "claim_id": spec.claim_id,
        "family": spec.family,
        "comparison_level": spec.comparison_level,
        "observability_class": spec.observability_class,
        "backend": backend,
        "api_surface": api_surface,
        "generator": generator,
        "distribution": distribution,
        "dtype": _dtype_name(distribution),
        "ordering": ordering,
        "seed": ctx.seed,
        "offset": ctx.offset,
        "N": int(n),
        "parameters": parameters or {},
        "timing_boundary": spec.timing_boundary,
        "validation": validation,
        "what_it_can_say": spec.can_say,
        "what_it_cannot_say": spec.cannot_say,
        "known_limitations": [spec.cannot_say],
    }


def _metadata_record(ctx: BenchmarkContext, spec: TaskSpec, backend: str, payload: dict[str, Any]) -> dict[str, Any]:
    record = _base_record(ctx, spec, backend, "metadata", "not_applicable", "metadata", 0, validation=validation_pass({"metadata_recorded": True}))
    record["payload"] = payload
    record["formal_result"] = True
    return record


def _metadata_rows(ctx: BenchmarkContext, spec: TaskSpec, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [_metadata_record(ctx, spec, "capability_matrix", payload)]
    for generator, info in payload.get("generators", {}).items():
        row = _metadata_record(ctx, spec, f"capability_{generator}", info)
        row["generator"] = generator
        rows.append(row)
    return rows


def _gate_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    backend: str,
    generator: str,
    distribution: str,
    n: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    record = _base_record(ctx, spec, backend, "gate", generator, distribution, n, validation=validation)
    record["formal_result"] = validation.get("status") == "pass"
    return record


def _unsupported_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    backend: str,
    reason: str,
    *,
    generator: str = "not_applicable",
    distribution: str = "not_applicable",
    n: int = 0,
    parameters: dict[str, Any] | None = None,
    comparison_key: str | None = None,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    record = _base_record(ctx, spec, backend, "unsupported", generator, distribution, n, validation=unsupported(reason), parameters=parameters)
    record.update(
        {
            "comparison_key": comparison_key,
            "is_baseline": False,
            "baseline_id": baseline_id,
            "formal_result": False,
            "audit_flags": ["unsupported_backend"],
        }
    )
    return record


def _error_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    backend: str,
    exc: BaseException,
    *,
    generator: str = "not_applicable",
    distribution: str = "not_applicable",
    n: int = 0,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = _base_record(ctx, spec, backend, "error", generator, distribution, n, validation=validation_error(exc), parameters=parameters)
    record.update({"formal_result": False, "audit_flags": ["runtime_error"]})
    return record


def finalize_records(records: list[dict[str, Any]]) -> None:
    apply_cross_record_gates(records)
    _compute_speedups(records)


def apply_cross_record_gates(records: list[dict[str, Any]]) -> None:
    failed_sequence: set[tuple[str, str]] = set()
    failed_distribution: set[tuple[str, str, str]] = set()
    failed_basic: set[tuple[str, str, str]] = set()
    for record in records:
        if record.get("validation", {}).get("status") != "fail":
            continue
        key2 = (str(record.get("backend")), str(record.get("generator")))
        key3 = (str(record.get("backend")), str(record.get("generator")), str(record.get("distribution")))
        if record.get("task_id") == "G2_REPRODUCIBILITY":
            failed_sequence.add(key2)
        elif record.get("task_id") == "G1_DISTRIBUTION_ROUGH_CHECK":
            failed_distribution.add(key3)
        elif record.get("task_id") == "G0_BASIC_CONTRACT":
            failed_basic.add(key3)

    gate_task_ids = {"G0_BASIC_CONTRACT", "G1_DISTRIBUTION_ROUGH_CHECK", "G2_REPRODUCIBILITY"}
    for record in records:
        if record.get("task_id") in gate_task_ids or not record.get("comparison_key"):
            continue
        key2 = (str(record.get("backend")), str(record.get("generator")))
        key3 = (str(record.get("backend")), str(record.get("generator")), str(record.get("distribution")))
        failures: list[str] = []
        if key3 in failed_basic:
            failures.append("G0_BASIC_CONTRACT")
            _add_audit_flag(record, "basic_contract_gate_failed")
        if key3 in failed_distribution:
            failures.append("G1_DISTRIBUTION_ROUGH_CHECK")
            _add_audit_flag(record, "distribution_gate_failed")
        if key2 in failed_sequence:
            failures.append("G2_REPRODUCIBILITY")
            _add_audit_flag(record, "sequence_semantics_gate_failed")
        if failures:
            record["formal_result"] = False
            record["cross_record_gate_failures"] = sorted(set(failures))


def _add_audit_flag(record: dict[str, Any], flag: str) -> None:
    flags = record.get("audit_flags")
    if not isinstance(flags, list):
        flags = []
        record["audit_flags"] = flags
    if flag not in flags:
        flags.append(flag)


def _compute_speedups(records: list[dict[str, Any]]) -> None:
    baselines: dict[str, dict[str, float]] = {}
    seen_baseline_keys: set[str] = set()
    for record in records:
        record["speedup_gpu_vs_baseline"] = None
        record["speedup_wall_vs_baseline"] = None
        record["speedup_baseline_formal"] = False
    for record in records:
        key = record.get("comparison_key")
        if not key or not record.get("is_baseline"):
            continue
        seen_baseline_keys.add(key)
        if not record.get("formal_result"):
            continue
        gpu = record.get("median_gpu_us")
        wall = record.get("median_wall_sync_us")
        if gpu is not None or wall is not None:
            baselines[key] = {"gpu": gpu, "wall": wall}
    for record in records:
        key = record.get("comparison_key")
        base = baselines.get(key)
        record["speedup_baseline_formal"] = bool(base)
        if key and base and not record.get("formal_result"):
            if not record.get("is_baseline"):
                _add_audit_flag(record, "record_not_formal")
            continue
        if not key or not base:
            if key in seen_baseline_keys and not record.get("is_baseline"):
                _add_audit_flag(record, "baseline_not_formal")
            continue
        gpu = record.get("median_gpu_us")
        wall = record.get("median_wall_sync_us")
        record["speedup_gpu_vs_baseline"] = _ratio(base.get("gpu"), gpu)
        record["speedup_wall_vs_baseline"] = _ratio(base.get("wall"), wall)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _estimate_output_bytes(distribution: str, n: int) -> int | None:
    if "raw64" in distribution:
        return n * 8
    if "dropout" in distribution:
        return n * 5
    if "threshold" in distribution:
        return n
    if distribution == "metadata":
        return None
    return n * 4


def _dtype_name(distribution: str) -> str:
    if "raw64" in distribution:
        return "uint64/int64_storage"
    if "raw32" in distribution or "poisson" in distribution:
        return "uint32/int32_storage"
    if "threshold" in distribution or "mask" in distribution:
        return "uint8"
    if distribution in {"metadata", "lifecycle", "generate_seeds", "error_case"}:
        return "not_applicable"
    return "float32"


def _e0_curand_invalid_lambda(ctx: BenchmarkContext) -> dict[str, Any]:
    out = torch.empty(16, device=ctx.device, dtype=torch.int32)
    try:
        with make_curand_generator("philox4x32_10", seed=ctx.seed, ordering="legacy") as gen:
            gen.generate_poisson_u32(out, lambda_val=-1.0)
            torch.cuda.synchronize()
    except BaseException as exc:
        return {"raised": True, "error_type": type(exc).__name__, "error": str(exc)}
    return {"raised": False}


def _e0_curand_odd_normal_n(ctx: BenchmarkContext) -> dict[str, Any]:
    out = torch.empty(17, device=ctx.device, dtype=torch.float32)
    try:
        with make_curand_generator("philox4x32_10", seed=ctx.seed, ordering="legacy") as gen:
            gen.generate_normal_f32(out)
            torch.cuda.synchronize()
    except BaseException as exc:
        return {"raised": True, "error_type": type(exc).__name__, "error": str(exc)}
    return {"raised": False}


def _e0_curand_invalid_dimensions(ctx: BenchmarkContext) -> dict[str, Any]:
    try:
        with make_curand_generator("sobol32", seed=ctx.seed, dimensions=0):
            pass
    except BaseException as exc:
        return {"raised": True, "error_type": type(exc).__name__, "error": str(exc)}
    return {"raised": False}


def _e0_flagrand_invalid_lambda(ctx: BenchmarkContext) -> dict[str, Any]:
    out = torch.empty(16, device=ctx.device, dtype=torch.int32)
    try:
        gen = make_flagrand_generator("philox4x32_10", seed=ctx.seed)
        flagrand_generate_by_distribution(gen, out, "poisson_u32", lambda_val=-1.0)
        torch.cuda.synchronize()
    except BaseException as exc:
        return {"raised": True, "error_type": type(exc).__name__, "error": str(exc)}
    return {"raised": False}


def _e0_flagrand_odd_philox_n(ctx: BenchmarkContext) -> dict[str, Any]:
    out = torch.empty(17, device=ctx.device, dtype=torch.int32)
    try:
        gen = make_flagrand_generator("philox4x32_10", seed=ctx.seed)
        flagrand_generate_by_distribution(gen, out, "raw32")
        torch.cuda.synchronize()
    except BaseException as exc:
        return {"raised": True, "error_type": type(exc).__name__, "error": str(exc)}
    return {"raised": False}


def _git_info() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return {"commit": sha, "dirty": bool(status.strip()), "status_short": status.splitlines()[:20]}
    except Exception as exc:
        return {"error": str(exc)}
