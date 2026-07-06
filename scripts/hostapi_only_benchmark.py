from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import torch  # noqa: E402

from contract_benchmark.generator_registry import GENERATOR_INFOS, GeneratorInfo  # noqa: E402
from contract_benchmark.runtime_env import configure_writable_cache  # noqa: E402
from contract_benchmark.timing import collect_cuda_event_and_wall_us  # noqa: E402
from contract_benchmark.validation import (  # noqa: E402
    validate_lognormal,
    validate_normal,
    validate_poisson,
    validate_raw_tensor,
    validate_uniform,
)


@dataclass(frozen=True)
class HostApiProfile:
    name: str
    sizes: list[int]
    warmup: int
    repeats: int
    poisson_lambdas: list[float]


HOSTAPI_PROFILES = {
    "local_smoke": HostApiProfile(
        name="local_smoke",
        sizes=[1024, 16384, 65536],
        warmup=1,
        repeats=3,
        poisson_lambdas=[1.0, 10.0],
    ),
    "local": HostApiProfile(
        name="local",
        sizes=[1024, 4096, 65536, 1048576],
        warmup=3,
        repeats=10,
        poisson_lambdas=[0.1, 1.0, 10.0, 64.0],
    ),
    "h20": HostApiProfile(
        name="h20",
        sizes=[4096, 16384, 65536, 262144, 1048576, 4194304, 8388608],
        warmup=5,
        repeats=20,
        poisson_lambdas=[0.1, 1.0, 4.0, 10.0, 32.0, 64.0, 256.0, 1024.0, 10000.0],
    ),
}

DEFAULT_DISTRIBUTIONS = ["raw", "uniform", "normal", "lognormal", "poisson"]
DEFAULT_GENERATORS = list(GENERATOR_INFOS)
BENCHMARK_NAME = "hostapi_only"
RUNTIME_CACHE_ENV: dict[str, str] = {}


@dataclass(frozen=True)
class HostApiCase:
    case_index: int
    case_id: str
    generator: str
    distribution: str
    n: int
    dtype_name: str
    parameters: dict[str, Any]
    dimensions: int | None
    notes: list[str]

    @property
    def torch_dtype(self) -> torch.dtype:
        return _torch_dtype(self.dtype_name)

    def to_record(self) -> dict[str, Any]:
        return {
            "case_index": self.case_index,
            "case_id": self.case_id,
            "generator": self.generator,
            "distribution": self.distribution,
            "N": self.n,
            "dtype": self.dtype_name,
            "parameters": self.parameters,
            "dimensions": self.dimensions,
            "notes": self.notes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host-API-only FlagRand vs cuRAND benchmark. This bypasses the task/gate benchmark.",
    )
    parser.add_argument("--profile", default=os.environ.get("PROFILE", "local_smoke"))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    parser.add_argument("--generators", default=os.environ.get("HOSTAPI_GENERATORS", "all"))
    parser.add_argument("--distributions", default=os.environ.get("HOSTAPI_DISTRIBUTIONS", "all"))
    parser.add_argument("--sizes", default=os.environ.get("HOSTAPI_SIZES", "profile"))
    parser.add_argument("--poisson-lambdas", default=os.environ.get("HOSTAPI_POISSON_LAMBDAS", "profile"))
    parser.add_argument("--warmup", type=int, default=_optional_int_env("HOSTAPI_WARMUP"))
    parser.add_argument("--repeats", type=int, default=_optional_int_env("HOSTAPI_REPEATS"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("HOSTAPI_SEED", "12345")))
    parser.add_argument("--offset", type=int, default=int(os.environ.get("HOSTAPI_OFFSET", "0")))
    parser.add_argument("--ordering", default=os.environ.get("HOSTAPI_ORDERING", "legacy"))
    parser.add_argument("--qrng-dimensions", type=int, default=int(os.environ.get("HOSTAPI_QRNG_DIMENSIONS", "1")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("HOSTAPI_RESULTS_DIR", REPO_ROOT / "results" / "hostapi_only")))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=_optional_int_env("HOSTAPI_MAX_CASES"))
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    started_unix = time.time()
    args = parse_args()
    profile = _profile_from_args(args)
    cases = build_cases(args, profile)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if args.list_cases:
        for case in cases:
            print(json.dumps(case.to_record(), ensure_ascii=False, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for hostapi-only benchmark.")
    if not cases:
        raise SystemExit("No hostapi-only benchmark cases selected.")

    if args.num_gpus > 1 and args.shard_index is None:
        return run_parallel(args, profile, cases, started_unix)
    return run_single(args, profile, cases, started_unix)


def build_cases(args: argparse.Namespace, profile: HostApiProfile) -> list[HostApiCase]:
    generators = _selected_generators(args.generators)
    requested_distributions = _selected_distributions(args.distributions)
    sizes = _selected_ints(args.sizes, profile.sizes)
    poisson_lambdas = _selected_floats(args.poisson_lambdas, profile.poisson_lambdas)

    cases: list[HostApiCase] = []
    case_index = 0
    for generator in generators:
        info = GENERATOR_INFOS[generator]
        if not info.supports_curand_host or not info.supports_flagrand:
            continue
        dimensions = args.qrng_dimensions if info.kind == "qrng" else None
        for distribution in _expand_distributions(info, requested_distributions):
            lambda_values = poisson_lambdas if distribution == "poisson_u32" else [None]
            for n0 in sizes:
                for lambda_val in lambda_values:
                    parameters: dict[str, Any] = {}
                    notes = list(info.notes)
                    if lambda_val is not None:
                        parameters["lambda"] = float(lambda_val)
                        if float(lambda_val) >= 30.0:
                            notes.append("FlagRand large-lambda Poisson uses the repository approximation path.")
                    n = _adjust_n(int(n0), generator, distribution, parameters)
                    dtype_name = _dtype_name_for_distribution(distribution)
                    case_id = _case_id(generator, distribution, n, parameters, dimensions)
                    cases.append(
                        HostApiCase(
                            case_index=case_index,
                            case_id=case_id,
                            generator=generator,
                            distribution=distribution,
                            n=n,
                            dtype_name=dtype_name,
                            parameters=parameters,
                            dimensions=dimensions,
                            notes=notes,
                        )
                    )
                    case_index += 1
    return cases


def run_parallel(
    args: argparse.Namespace,
    profile: HostApiProfile,
    cases: list[HostApiCase],
    started_unix: float,
) -> int:
    gpu_ids = _visible_gpu_ids(args.num_gpus)
    shard_count = min(max(1, args.num_gpus), len(gpu_ids), len(cases))
    gpu_ids = gpu_ids[:shard_count]

    root_dir = args.run_dir or _make_run_dir(args.results_dir, f"{profile.name}_hostapi_only_parallel_{shard_count}gpu")
    root_dir.mkdir(parents=True, exist_ok=False)
    _configure_runtime_cache(root_dir)
    manifest: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "profile": profile.name,
        "requested_num_gpus": args.num_gpus,
        "shard_count": shard_count,
        "gpu_ids": gpu_ids,
        "root_dir": str(root_dir),
        "case_count": len(cases),
        "started_unix": started_unix,
        "shards": [],
    }
    print(f"[hostapi-only] results: {root_dir}", flush=True)
    print(f"[hostapi-only] cases={len(cases)} shards={shard_count} gpu_ids={gpu_ids}", flush=True)

    processes: list[tuple[int, Path, subprocess.Popen[bytes], Any, dict[str, Any]]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        run_dir = root_dir / f"shard_{shard_index:02d}_gpu_{_safe_name(gpu_id)}"
        log_path = root_dir / f"shard_{shard_index:02d}.log"
        command = _child_command(args, run_dir, shard_index, shard_count)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["CURAND_CONTRACT_SHARD"] = str(shard_index)
        env["CURAND_CONTRACT_SHARD_COUNT"] = str(shard_count)
        env["HOME"] = str(root_dir)
        env["XDG_CACHE_HOME"] = str(root_dir / ".cache" / f"shard_{shard_index:02d}")
        env["TRITON_CACHE_DIR"] = str(root_dir / ".cache" / "triton" / f"shard_{shard_index:02d}")
        env["TORCH_EXTENSIONS_DIR"] = str(root_dir / ".cache" / "torch_extensions")
        Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
        shard_manifest = {
            "shard": shard_index,
            "gpu": gpu_id,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "launch_started_unix": time.time(),
        }
        manifest["shards"].append(shard_manifest)
        log_file = log_path.open("wb")
        print(f"[hostapi-only] launch shard={shard_index} gpu={gpu_id} log={log_path}", flush=True)
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((shard_index, run_dir, process, log_file, shard_manifest))

    _write_json(root_dir / "parallel_manifest.json", manifest)

    failures: list[str] = []
    for shard_index, run_dir, process, log_file, shard_manifest in processes:
        rc = process.wait()
        ended_unix = time.time()
        log_file.close()
        shard_manifest["exit_code"] = rc
        shard_manifest["ended_unix"] = ended_unix
        shard_manifest["elapsed_seconds"] = ended_unix - float(shard_manifest["launch_started_unix"])
        if rc:
            failures.append(f"shard {shard_index} failed with exit code {rc}; see {root_dir / f'shard_{shard_index:02d}.log'}")
        else:
            print(f"[hostapi-only] shard={shard_index} done elapsed_seconds={shard_manifest['elapsed_seconds']:.3f}", flush=True)

    records: list[dict[str, Any]] = []
    shard_environments: list[dict[str, Any]] = []
    for _, run_dir, _, _, _ in processes:
        if (run_dir / "environment.json").exists():
            shard_environments.append(json.loads((run_dir / "environment.json").read_text(encoding="utf-8")))
        if (run_dir / "records.jsonl").exists():
            records.extend(_read_jsonl(run_dir / "records.jsonl"))

    _add_speedups(records)
    ended_unix = time.time()
    environment = collect_host_environment(
        profile,
        args,
        started_unix=started_unix,
        ended_unix=ended_unix,
        extra={"parallel_launcher": {"shards": manifest["shards"], "failures": failures, "shard_environments": shard_environments}},
    )
    summary = summarize_records(records, cases, environment, failures=failures)
    manifest["ended_unix"] = ended_unix
    manifest["elapsed_seconds"] = ended_unix - started_unix
    manifest["record_count"] = len(records)
    _write_outputs(root_dir, records, cases, environment, summary, manifest)
    _print_summary("[hostapi-only]", root_dir, summary)
    return 1 if failures else 0


def run_single(
    args: argparse.Namespace,
    profile: HostApiProfile,
    cases: list[HostApiCase],
    started_unix: float,
) -> int:
    if args.shard_index is not None:
        shard_count = int(args.shard_count or 1)
        cases = [case for case in cases if case.case_index % shard_count == args.shard_index]
    run_dir = args.run_dir or _make_run_dir(args.results_dir, f"{profile.name}_hostapi_only")
    run_dir.mkdir(parents=True, exist_ok=False)
    _configure_runtime_cache(run_dir, shard=args.shard_index)
    print(f"[hostapi-only] results: {run_dir}", flush=True)
    print(f"[hostapi-only] profile={profile.name} cases={len(cases)} warmup={_warmup(args, profile)} repeats={_repeats(args, profile)}", flush=True)

    records: list[dict[str, Any]] = []
    for local_index, case in enumerate(cases, start=1):
        print(
            f"[hostapi-only] case {local_index}/{len(cases)} "
            f"{case.generator} {case.distribution} n={case.n} params={case.parameters}",
            flush=True,
        )
        records.extend(run_case(case, args=args, profile=profile))

    _add_speedups(records)
    ended_unix = time.time()
    environment = collect_host_environment(profile, args, started_unix=started_unix, ended_unix=ended_unix)
    summary = summarize_records(records, cases, environment, failures=[])
    manifest = {
        "benchmark": BENCHMARK_NAME,
        "profile": profile.name,
        "root_dir": str(run_dir),
        "case_count": len(cases),
        "record_count": len(records),
        "started_unix": started_unix,
        "ended_unix": ended_unix,
        "elapsed_seconds": ended_unix - started_unix,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    _write_outputs(run_dir, records, cases, environment, summary, manifest)
    _print_summary("[hostapi-only]", run_dir, summary)
    return 0


def run_case(case: HostApiCase, *, args: argparse.Namespace, profile: HostApiProfile) -> list[dict[str, Any]]:
    return [
        run_backend(case, "curand_host_api", args=args, profile=profile),
        run_backend(case, "flagrand_public_api", args=args, profile=profile),
    ]


def run_backend(
    case: HostApiCase,
    backend: str,
    *,
    args: argparse.Namespace,
    profile: HostApiProfile,
) -> dict[str, Any]:
    out = torch.empty(case.n, device="cuda", dtype=case.torch_dtype)
    gen: Any | None = None
    try:
        if backend == "curand_host_api":
            from contract_benchmark.curand_adapter import curand_generate_by_distribution, make_curand_generator

            gen = make_curand_generator(
                case.generator,
                seed=args.seed,
                offset=args.offset,
                ordering=args.ordering,
                dimensions=case.dimensions,
            )
            run_once = lambda: curand_generate_by_distribution(gen, out, case.distribution, **_distribution_kwargs(case))
        elif backend == "flagrand_public_api":
            from contract_benchmark.flagrand_adapter import flagrand_generate_by_distribution, make_flagrand_generator

            gen = make_flagrand_generator(
                case.generator,
                seed=args.seed,
                offset=args.offset,
                dimensions=case.dimensions,
            )
            run_once = lambda: flagrand_generate_by_distribution(gen, out, case.distribution, **_distribution_kwargs(case))
        else:
            raise ValueError(f"unknown backend={backend}")

        validation = _validate_after_run(run_once, out, case)
        timing = collect_cuda_event_and_wall_us(
            run_once,
            warmup_iters=_warmup(args, profile),
            repeats=_repeats(args, profile),
        )
        return _timed_record(case, backend, timing, validation, args=args)
    except BaseException as exc:
        return _exception_record(case, backend, exc, args=args)
    finally:
        if backend == "curand_host_api" and gen is not None:
            try:
                gen.destroy()
            except BaseException:
                pass


def _validate_after_run(run_once: Any, out: torch.Tensor, case: HostApiCase) -> dict[str, Any]:
    try:
        run_once()
        torch.cuda.synchronize()
        if case.distribution in {"raw32", "raw64"}:
            return validate_raw_tensor(out, dtype=case.torch_dtype, n=case.n)
        if case.distribution in {"uniform_f32", "uniform_f64"}:
            return validate_uniform(out, n=case.n, low_open=True)
        if case.distribution in {"normal_f32", "normal_f64"}:
            return validate_normal(out, n=case.n, mean=0.0, stddev=1.0)
        if case.distribution in {"lognormal_f32", "lognormal_f64"}:
            return validate_lognormal(out, n=case.n)
        if case.distribution == "poisson_u32":
            return validate_poisson(out, n=case.n, lambda_val=float(case.parameters["lambda"]))
        return {"status": "pass", "checks": {"shape_numel": out.numel() == case.n}}
    except BaseException as exc:
        return {"status": "fail", "error_type": type(exc).__name__, "error": str(exc)}


def _timed_record(
    case: HostApiCase,
    backend: str,
    timing: Any,
    validation: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    timing_record = timing.to_record()
    median_gpu_us = timing_record.get("median_gpu_us")
    median_wall_us = timing_record.get("median_wall_sync_us")
    output_bytes = case.n * torch.empty((), dtype=case.torch_dtype).element_size()
    return {
        **case.to_record(),
        "benchmark": BENCHMARK_NAME,
        "backend": backend,
        "is_baseline": backend == "curand_host_api",
        "ordering": args.ordering if GENERATOR_INFOS[case.generator].kind == "prng" else None,
        "seed": args.seed if GENERATOR_INFOS[case.generator].supports_seed else None,
        "offset": args.offset,
        "status": "ok" if validation.get("status") == "pass" else "validation_fail",
        "validation": validation,
        "output_bytes": output_bytes,
        "temporary_bytes": _temporary_bytes(case, backend),
        "timing": timing_record,
        "median_gpu_us": median_gpu_us,
        "median_wall_sync_us": median_wall_us,
        "median_cpu_enqueue_us": timing_record.get("median_cpu_enqueue_us"),
        "items_per_second_gpu": _rate(case.n, median_gpu_us),
        "gib_per_second_gpu": _rate(output_bytes / (1024.0**3), median_gpu_us),
        "items_per_second_wall": _rate(case.n, median_wall_us),
    }


def _exception_record(
    case: HostApiCase,
    backend: str,
    exc: BaseException,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        **case.to_record(),
        "benchmark": BENCHMARK_NAME,
        "backend": backend,
        "is_baseline": backend == "curand_host_api",
        "ordering": args.ordering if GENERATOR_INFOS[case.generator].kind == "prng" else None,
        "seed": args.seed if GENERATOR_INFOS[case.generator].supports_seed else None,
        "offset": args.offset,
        "status": "unsupported",
        "validation": {
            "status": "unsupported",
            "unsupported_reason": str(exc),
            "error_type": type(exc).__name__,
        },
        "output_bytes": case.n * torch.empty((), dtype=case.torch_dtype).element_size(),
        "temporary_bytes": _temporary_bytes(case, backend),
        "timing": {},
        "median_gpu_us": None,
        "median_wall_sync_us": None,
        "median_cpu_enqueue_us": None,
        "items_per_second_gpu": None,
        "gib_per_second_gpu": None,
        "items_per_second_wall": None,
    }


def summarize_records(
    records: list[dict[str, Any]],
    cases: list[HostApiCase],
    environment: dict[str, Any],
    *,
    failures: list[str],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    backend_status_counts: dict[str, dict[str, int]] = {}
    for record in records:
        status = str(record.get("status"))
        backend = str(record.get("backend"))
        status_counts[status] = status_counts.get(status, 0) + 1
        backend_status_counts.setdefault(backend, {})
        backend_status_counts[backend][status] = backend_status_counts[backend].get(status, 0) + 1

    paired = _paired_success_records(records)
    speedups = [
        float(record["speedup_gpu_vs_curand_host"])
        for record in paired
        if record.get("backend") == "flagrand_public_api" and record.get("speedup_gpu_vs_curand_host") is not None
    ]
    return {
        "benchmark": BENCHMARK_NAME,
        "profile": environment.get("profile"),
        "case_count": len(cases),
        "record_count": len(records),
        "paired_success_case_count": len({record["case_id"] for record in paired}) if paired else 0,
        "status_counts": status_counts,
        "backend_status_counts": backend_status_counts,
        "speedup_gpu_vs_curand_host": _sample_summary(speedups),
        "failures": failures,
        "run_health": {
            "status": "ok" if not failures else "needs_attention",
            "shard_process_failure_count": len(failures),
            "unsupported_record_count": status_counts.get("unsupported", 0),
            "validation_fail_record_count": status_counts.get("validation_fail", 0),
        },
    }


def _add_speedups(records: list[dict[str, Any]]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(str(record["case_id"]), {})[str(record["backend"])] = record
    for backends in by_case.values():
        baseline = backends.get("curand_host_api")
        if not baseline or baseline.get("status") != "ok":
            continue
        baseline_gpu = baseline.get("median_gpu_us")
        baseline_wall = baseline.get("median_wall_sync_us")
        for record in backends.values():
            record["curand_host_median_gpu_us"] = baseline_gpu
            record["curand_host_median_wall_sync_us"] = baseline_wall
            record["speedup_gpu_vs_curand_host"] = _speedup(baseline_gpu, record.get("median_gpu_us"))
            record["speedup_wall_vs_curand_host"] = _speedup(baseline_wall, record.get("median_wall_sync_us"))


def _paired_success_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(str(record["case_id"]), {})[str(record["backend"])] = record
    paired: list[dict[str, Any]] = []
    for backends in by_case.values():
        curand = backends.get("curand_host_api")
        flagrand = backends.get("flagrand_public_api")
        if curand and flagrand and curand.get("status") == "ok" and flagrand.get("status") == "ok":
            paired.extend([curand, flagrand])
    return paired


def collect_host_environment(
    profile: HostApiProfile,
    args: argparse.Namespace,
    *,
    started_unix: float,
    ended_unix: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "profile": profile.name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_from_torch": torch.version.cuda,
        "runtime_cache": RUNTIME_CACHE_ENV,
        "run_timing": {
            "started_unix": started_unix,
            "ended_unix": ended_unix,
            "elapsed_seconds": ended_unix - started_unix,
        },
        "args": _serializable_args(args),
        "git": _git_info(),
    }
    if torch.cuda.is_available():
        env.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    try:
        from contract_benchmark.curand_library import library_load_report

        env["curand"] = library_load_report()
    except BaseException as exc:
        env["curand"] = {"available": False, "error": str(exc), "error_type": type(exc).__name__}
    if extra:
        env.update(extra)
    return env


def _write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    cases: list[HostApiCase],
    environment: dict[str, Any],
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "case_matrix.json", [case.to_record() for case in cases])
    _write_jsonl(run_dir / "records.jsonl", records)
    _write_csv(run_dir / "records.csv", records)
    _write_json(run_dir / "summary.json", summary)
    _write_report(run_dir / "REPORT.md", records, summary, environment)


def _configure_runtime_cache(run_dir: Path, *, shard: int | None = None) -> None:
    global RUNTIME_CACHE_ENV
    RUNTIME_CACHE_ENV = configure_writable_cache(run_dir, shard=shard)


def _write_report(path: Path, records: list[dict[str, Any]], summary: dict[str, Any], environment: dict[str, Any]) -> None:
    paired_flagrand = [
        record for record in records
        if record.get("backend") == "flagrand_public_api" and record.get("speedup_gpu_vs_curand_host") is not None
    ]
    largest_n = max((int(record["N"]) for record in paired_flagrand), default=None)
    largest_rows = [record for record in paired_flagrand if largest_n is not None and int(record["N"]) == largest_n]
    worst_rows = sorted(paired_flagrand, key=lambda r: float(r.get("speedup_gpu_vs_curand_host") or 0.0))[:20]
    best_rows = sorted(paired_flagrand, key=lambda r: float(r.get("speedup_gpu_vs_curand_host") or 0.0), reverse=True)[:20]

    lines = [
        "# Host API Only Benchmark Report",
        "",
        "This benchmark compares only two surfaces: cuRAND Host API as the baseline and FlagRand public API as the candidate.",
        "It does not run the contract task registry, Device API extension, cuRANDDx extension, fused-consume baselines, or gate logic.",
        "",
        "## Environment",
        "",
        f"- profile: `{environment.get('profile')}`",
        f"- torch: `{environment.get('torch_version')}`",
        f"- cuda available: `{environment.get('cuda_available')}`",
        f"- gpu: `{environment.get('gpu_name')}`",
        f"- cuRAND version: `{environment.get('curand', {}).get('version')}`",
        f"- git commit: `{environment.get('git', {}).get('commit')}`",
        "",
        "## Summary",
        "",
        f"- cases: `{summary.get('case_count')}`",
        f"- records: `{summary.get('record_count')}`",
        f"- paired successful cases: `{summary.get('paired_success_case_count')}`",
        f"- run health: `{summary.get('run_health', {}).get('status')}`",
        f"- status counts: `{json.dumps(summary.get('status_counts', {}), sort_keys=True)}`",
        f"- speedup gpu vs cuRAND Host median summary: `{json.dumps(summary.get('speedup_gpu_vs_curand_host', {}), sort_keys=True)}`",
        "",
        "## Largest-N Paired Results",
        "",
    ]
    lines.extend(_markdown_table(largest_rows, limit=80))
    lines.extend(["", "## Slowest FlagRand Relative Cases", ""])
    lines.extend(_markdown_table(worst_rows, limit=20))
    lines.extend(["", "## Fastest FlagRand Relative Cases", ""])
    lines.extend(_markdown_table(best_rows, limit=20))
    unsupported = [record for record in records if record.get("status") == "unsupported"]
    if unsupported:
        lines.extend(["", "## Unsupported Records", ""])
        lines.extend(_unsupported_table(unsupported, limit=40))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `speedup_gpu_vs_curand_host > 1` means FlagRand was faster than cuRAND Host API by CUDA-event median.",
            "- Large-lambda FlagRand Poisson rows use the repository approximation path and are marked in case notes.",
            "- Full per-repeat timing samples are in `records.jsonl`; compact analysis rows are in `records.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(records: list[dict[str, Any]], *, limit: int) -> list[str]:
    if not records:
        return ["No paired rows."]
    lines = [
        "| generator | distribution | N | params | curand_gpu_us | flagrand_gpu_us | speedup_gpu | flagrand_wall_us |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for record in records[:limit]:
        lines.append(
            "| {generator} | {distribution} | {n} | `{params}` | {curand_gpu} | {flagrand_gpu} | {speedup} | {wall} |".format(
                generator=record["generator"],
                distribution=record["distribution"],
                n=record["N"],
                params=json.dumps(record.get("parameters", {}), sort_keys=True),
                curand_gpu=_fmt(record.get("curand_host_median_gpu_us")),
                flagrand_gpu=_fmt(record.get("median_gpu_us")),
                speedup=_fmt(record.get("speedup_gpu_vs_curand_host")),
                wall=_fmt(record.get("median_wall_sync_us")),
            )
        )
    if len(records) > limit:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... |")
    return lines


def _unsupported_table(records: list[dict[str, Any]], *, limit: int) -> list[str]:
    lines = [
        "| backend | generator | distribution | N | reason |",
        "|---|---|---:|---:|---|",
    ]
    for record in records[:limit]:
        validation = record.get("validation", {})
        reason = validation.get("unsupported_reason") or validation.get("error") or ""
        lines.append(
            f"| {record.get('backend')} | {record.get('generator')} | {record.get('distribution')} | {record.get('N')} | `{str(reason)[:160]}` |"
        )
    if len(records) > limit:
        lines.append("| ... | ... | ... | ... | ... |")
    return lines


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "case_index",
        "case_id",
        "backend",
        "is_baseline",
        "generator",
        "distribution",
        "N",
        "dtype",
        "parameters",
        "dimensions",
        "status",
        "validation_status",
        "unsupported_reason",
        "median_gpu_us",
        "median_wall_sync_us",
        "median_cpu_enqueue_us",
        "items_per_second_gpu",
        "gib_per_second_gpu",
        "items_per_second_wall",
        "speedup_gpu_vs_curand_host",
        "speedup_wall_vs_curand_host",
        "curand_host_median_gpu_us",
        "curand_host_median_wall_sync_us",
        "output_bytes",
        "temporary_bytes",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            validation = record.get("validation", {})
            row = {field: record.get(field) for field in fields}
            row["parameters"] = json.dumps(record.get("parameters", {}), ensure_ascii=False, sort_keys=True)
            row["notes"] = json.dumps(record.get("notes", []), ensure_ascii=False)
            row["validation_status"] = validation.get("status")
            row["unsupported_reason"] = validation.get("unsupported_reason")
            writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _print_summary(prefix: str, run_dir: Path, summary: dict[str, Any]) -> None:
    print(
        f"{prefix} records={summary.get('record_count')} cases={summary.get('case_count')} "
        f"paired={summary.get('paired_success_case_count')} status={summary.get('status_counts')}",
        flush=True,
    )
    print(f"{prefix} report: {run_dir / 'REPORT.md'}", flush=True)


def _profile_from_args(args: argparse.Namespace) -> HostApiProfile:
    profile = HOSTAPI_PROFILES.get(args.profile)
    if profile is None:
        choices = ", ".join(sorted(HOSTAPI_PROFILES))
        raise SystemExit(f"Unknown profile={args.profile!r}; expected one of {choices}")
    return profile


def _selected_generators(value: str) -> list[str]:
    tokens = _split_csv(value)
    if not tokens or tokens == ["all"]:
        return DEFAULT_GENERATORS
    unknown = [item for item in tokens if item not in GENERATOR_INFOS]
    if unknown:
        raise SystemExit(f"Unknown generator(s): {unknown}. Valid: {sorted(GENERATOR_INFOS)}")
    return tokens


def _selected_distributions(value: str) -> list[str]:
    tokens = _split_csv(value)
    return DEFAULT_DISTRIBUTIONS if not tokens or tokens == ["all"] else tokens


def _selected_ints(value: str, profile_values: list[int]) -> list[int]:
    if value in {"", "profile"}:
        return list(profile_values)
    return [int(item) for item in _split_csv(value)]


def _selected_floats(value: str, profile_values: list[float]) -> list[float]:
    if value in {"", "profile"}:
        return list(profile_values)
    return [float(item) for item in _split_csv(value)]


def _expand_distributions(info: GeneratorInfo, requested: list[str]) -> list[str]:
    distributions: list[str] = []
    for item in requested:
        if item == "raw":
            if info.supports_raw32:
                distributions.append("raw32")
            if info.supports_raw64:
                distributions.append("raw64")
        elif item == "uniform":
            distributions.append("uniform_f64" if info.supports_raw64 else "uniform_f32")
        elif item == "normal":
            distributions.append("normal_f64" if info.supports_raw64 else "normal_f32")
        elif item == "lognormal":
            distributions.append("lognormal_f64" if info.supports_raw64 else "lognormal_f32")
        elif item == "poisson":
            if info.kind == "prng" and info.supports_raw32:
                distributions.append("poisson_u32")
        else:
            if _distribution_supported_by_info(info, item):
                distributions.append(item)
    return list(dict.fromkeys(distributions))


def _distribution_supported_by_info(info: GeneratorInfo, distribution: str) -> bool:
    if distribution == "raw32":
        return info.supports_raw32
    if distribution == "raw64":
        return info.supports_raw64
    if distribution in {"uniform_f32", "normal_f32", "lognormal_f32"}:
        return info.supports_raw32
    if distribution in {"uniform_f64", "normal_f64", "lognormal_f64"}:
        return info.supports_raw64
    if distribution == "poisson_u32":
        return info.kind == "prng" and info.supports_raw32
    raise SystemExit(f"Unknown distribution selector: {distribution}")


def _adjust_n(n: int, generator: str, distribution: str, parameters: dict[str, Any]) -> int:
    value = max(1, int(n))
    if generator == "philox4x32_10":
        value += (-value % 4)
    if distribution in {"normal_f32", "normal_f64", "lognormal_f32", "lognormal_f64"} and value % 2:
        value += 1
    if distribution == "poisson_u32" and float(parameters.get("lambda", 0.0)) >= 30.0 and value % 2:
        value += 1
    return value


def _case_id(generator: str, distribution: str, n: int, parameters: dict[str, Any], dimensions: int | None) -> str:
    suffix = ""
    if "lambda" in parameters:
        suffix += f":lambda={parameters['lambda']:g}"
    if dimensions is not None:
        suffix += f":dim={dimensions}"
    return f"{generator}:{distribution}:n={n}{suffix}"


def _dtype_name_for_distribution(distribution: str) -> str:
    if distribution in {"raw32", "poisson_u32"}:
        return "int32"
    if distribution == "raw64":
        return "int64"
    if distribution.endswith("_f64"):
        return "float64"
    return "float32"


def _torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
        "float64": torch.float64,
    }[dtype_name]


def _temporary_bytes(case: HostApiCase, backend: str) -> int:
    if backend != "flagrand_public_api":
        return 0
    if case.distribution in {"uniform_f32", "normal_f32", "lognormal_f32", "poisson_u32"}:
        return case.n * torch.empty((), dtype=torch.int32).element_size()
    if case.distribution in {"uniform_f64", "normal_f64", "lognormal_f64"}:
        return case.n * torch.empty((), dtype=torch.int64).element_size()
    return 0


def _distribution_kwargs(case: HostApiCase) -> dict[str, Any]:
    if case.distribution == "poisson_u32":
        return {"lambda_val": float(case.parameters["lambda"])}
    return {}


def _rate(items: float, median_us: Any) -> float | None:
    if median_us is None or float(median_us) <= 0:
        return None
    return float(items) / (float(median_us) / 1_000_000.0)


def _speedup(baseline_us: Any, candidate_us: Any) -> float | None:
    if baseline_us is None or candidate_us is None:
        return None
    baseline = float(baseline_us)
    candidate = float(candidate_us)
    if baseline <= 0 or candidate <= 0:
        return None
    return baseline / candidate


def _sample_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": float(statistics.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _make_run_dir(base_dir: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{stamp}_{label}"


def _visible_gpu_ids(requested: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in {"NoDevFiles", ""}:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
    else:
        ids = [str(index) for index in range(torch.cuda.device_count())]
    return ids[: max(1, int(requested))]


def _child_command(args: argparse.Namespace, run_dir: Path, shard_index: int, shard_count: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile",
        args.profile,
        "--num-gpus",
        "1",
        "--generators",
        args.generators,
        "--distributions",
        args.distributions,
        "--sizes",
        args.sizes,
        "--poisson-lambdas",
        args.poisson_lambdas,
        "--seed",
        str(args.seed),
        "--offset",
        str(args.offset),
        "--ordering",
        args.ordering,
        "--qrng-dimensions",
        str(args.qrng_dimensions),
        "--run-dir",
        str(run_dir),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]
    if args.warmup is not None:
        command.extend(["--warmup", str(args.warmup)])
    if args.repeats is not None:
        command.extend(["--repeats", str(args.repeats)])
    if args.max_cases is not None:
        command.extend(["--max-cases", str(args.max_cases)])
    return command


def _warmup(args: argparse.Namespace, profile: HostApiProfile) -> int:
    return int(profile.warmup if args.warmup is None else args.warmup)


def _repeats(args: argparse.Namespace, profile: HostApiProfile) -> int:
    return int(profile.repeats if args.repeats is None else args.repeats)


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value) or "gpu"


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _git_info() -> dict[str, Any]:
    launcher_git_commit = os.environ.get("CURAND_CONTRACT_GIT_SHA")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL)
        return {"commit": sha, "dirty": bool(status.strip()), "status_short": status.splitlines()[:20], "source": "git"}
    except BaseException as exc:
        info: dict[str, Any] = {"error": str(exc)}
        if launcher_git_commit:
            info.update({"commit": launcher_git_commit, "dirty": None, "status_short": [], "source": "CURAND_CONTRACT_GIT_SHA"})
        return info


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
