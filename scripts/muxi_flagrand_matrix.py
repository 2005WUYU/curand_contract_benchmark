from __future__ import annotations

import argparse
import csv
import ctypes.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import torch  # noqa: E402
import triton  # noqa: E402
from triton.runtime.driver import driver  # noqa: E402

from contract_benchmark.flagrand_provenance import require_vendored_flagrand  # noqa: E402
from flagrand import curand  # noqa: E402


@dataclass(frozen=True)
class Profile:
    sizes: tuple[int, ...]
    warmup: int
    repeats: int
    target_items_per_batch: int
    max_calls_per_batch: int


PROFILES = {
    "smoke": Profile(
        sizes=(4096, 65536, 1048576),
        warmup=2,
        repeats=3,
        target_items_per_batch=1 << 22,
        max_calls_per_batch=128,
    ),
    "full": Profile(
        sizes=(4096, 16384, 65536, 262144, 1048576, 4194304, 8388608),
        warmup=5,
        repeats=9,
        target_items_per_batch=1 << 26,
        max_calls_per_batch=1000,
    ),
}


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    rng_type: str
    distributions: tuple[str, ...]


def generator_specs() -> tuple[GeneratorSpec, ...]:
    pseudo_distributions = (
        "raw32",
        "uniform_f32",
        "normal_f32",
        "lognormal_f32",
        "poisson_u32",
    )
    quasi32_distributions = (
        "raw32",
        "uniform_f32",
        "normal_f32",
        "lognormal_f32",
    )
    quasi64_distributions = (
        "raw64",
        "uniform_f64",
        "normal_f64",
        "lognormal_f64",
    )
    return (
        GeneratorSpec("philox4x32_10", curand.CURAND_RNG_PSEUDO_PHILOX4_32_10, pseudo_distributions),
        GeneratorSpec("xorwow", curand.CURAND_RNG_PSEUDO_XORWOW, pseudo_distributions),
        GeneratorSpec("mrg32k3a", curand.CURAND_RNG_PSEUDO_MRG32K3A, pseudo_distributions),
        GeneratorSpec("mtgp32", curand.CURAND_RNG_PSEUDO_MTGP32, pseudo_distributions),
        GeneratorSpec("mt19937", curand.CURAND_RNG_PSEUDO_MT19937, pseudo_distributions),
        GeneratorSpec("sobol32", curand.CURAND_RNG_QUASI_SOBOL32, quasi32_distributions),
        GeneratorSpec(
            "scrambled_sobol32",
            curand.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
            quasi32_distributions,
        ),
        GeneratorSpec("sobol64", curand.CURAND_RNG_QUASI_SOBOL64, quasi64_distributions),
        GeneratorSpec(
            "scrambled_sobol64",
            curand.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
            quasi64_distributions,
        ),
    )


DTYPES = {
    "raw32": torch.int32,
    "raw64": torch.int64,
    "uniform_f32": torch.float32,
    "uniform_f64": torch.float64,
    "normal_f32": torch.float32,
    "normal_f64": torch.float64,
    "lognormal_f32": torch.float32,
    "lognormal_f64": torch.float64,
    "poisson_u32": torch.int32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MetaX/MXMACA FlagRand correctness and performance matrix"
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument(
        "--generators",
        default="all",
        help="Comma-separated generators or all.",
    )
    parser.add_argument(
        "--distributions",
        default="all",
        help="Comma-separated distributions or all.",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Optional comma-separated element counts overriding the profile.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dimensions", type=int, default=1)
    parser.add_argument("--poisson-lambda", type=float, default=4.0)
    parser.add_argument("--skip-torch-reference", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "muxi_results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A torch CUDA-compatible accelerator is required.")

    vendor_report = require_vendored_flagrand()
    compatibility = apply_maca_compatibility()
    profile = PROFILES[args.profile]
    sizes = parse_int_list(args.sizes) if args.sizes else profile.sizes
    selected_generators = parse_name_set(args.generators)
    selected_distributions = parse_name_set(args.distributions)

    run_dir = make_run_dir(args.results_dir, args.profile)
    environment = collect_environment(args, profile, sizes, compatibility, vendor_report)
    write_json(run_dir / "environment.json", environment)

    print(f"[muxi-matrix] results={run_dir}")
    print(
        "[muxi-matrix] "
        f"device={environment['device']['name']} target={environment['triton_target']} "
        f"profile={args.profile} sizes={list(sizes)}"
    )
    print(
        "[muxi-matrix] cuRAND=unavailable; "
        "torch RNG is a platform reference, not an algorithm-equivalent baseline"
    )

    records: list[dict[str, Any]] = []
    for spec in generator_specs():
        if selected_generators is not None and spec.name not in selected_generators:
            continue
        for distribution in spec.distributions:
            if selected_distributions is not None and distribution not in selected_distributions:
                continue
            for n in sizes:
                record = run_flagrand_case(
                    spec,
                    distribution,
                    n,
                    args=args,
                    profile=profile,
                )
                records.append(record)
                print_record(record)

    if not args.skip_torch_reference:
        for n in sizes:
            records.extend(run_torch_references(n, profile))
            for record in records[-10:]:
                if record.get("n") == n and record.get("backend") != "flagrand_public":
                    print_record(record)

    summary = summarize(records, environment)
    write_jsonl(run_dir / "results.jsonl", records)
    write_csv(run_dir / "results.csv", records)
    write_json(run_dir / "summary.json", summary)
    write_report(run_dir / "REPORT.md", records, summary, environment)

    print(
        "[muxi-matrix] "
        f"pass={summary['status_counts'].get('pass', 0)} "
        f"error={summary['status_counts'].get('error', 0)} "
        f"validation_fail={summary['status_counts'].get('validation_fail', 0)}"
    )
    print(f"[muxi-matrix] report={run_dir / 'REPORT.md'}")
    return 0 if summary["run_health"] == "ok" else 1


def apply_maca_compatibility() -> dict[str, Any]:
    target = driver.active.get_current_target()
    backend = str(getattr(target, "backend", "unknown"))
    applied: list[str] = []
    if backend != "maca":
        return {"backend": backend, "applied": applied}

    import flagrand.runtime.compiled_launcher as compiled_launcher
    from flagrand.fused import lognormal as fused_lognormal
    from flagrand.fused import normal as fused_normal
    from flagrand.fused import poisson as fused_poisson
    from flagrand.fused._internal import philox_direct, state_prng_kernels, transforms
    from flagrand.rng import _stateful_output

    compiled_launcher._requires_regular_jit = lambda kernel: True
    applied.append("standard_triton_jit_launcher")

    portable = transforms.uniform_to_normal_trig
    philox_direct.uniform_to_normal_fast_f32 = portable
    state_prng_kernels.uniform_to_normal = portable
    _stateful_output.uniform_to_normal_fast_f32 = portable
    fused_normal.uniform_to_normal = portable
    fused_lognormal.uniform_to_normal = portable
    fused_poisson.uniform_to_normal = portable
    applied.append("portable_tl_sin_cos_box_muller_f32")
    return {"backend": backend, "applied": applied}


def run_flagrand_case(
    spec: GeneratorSpec,
    distribution: str,
    n: int,
    *,
    args: argparse.Namespace,
    profile: Profile,
) -> dict[str, Any]:
    record = base_record(
        backend="flagrand_public",
        generator=spec.name,
        distribution=distribution,
        n=n,
        dtype=DTYPES[distribution],
    )
    record["parameters"] = {
        "seed": args.seed,
        "dimensions": args.dimensions,
        "poisson_lambda": args.poisson_lambda if distribution == "poisson_u32" else None,
    }
    try:
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        generator = make_generator(spec, args.seed, args.dimensions)
        invoke_flagrand(distribution, generator, output, args.poisson_lambda)
        torch.cuda.synchronize()

        replay = torch.empty_like(output)
        replay_generator = make_generator(spec, args.seed, args.dimensions)
        invoke_flagrand(distribution, replay_generator, replay, args.poisson_lambda)
        torch.cuda.synchronize()

        validation = validate_output(
            output,
            replay,
            distribution,
            poisson_lambda=args.poisson_lambda,
        )
        record["validation"] = validation
        if validation["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = make_generator(spec, args.seed, args.dimensions)
        timing = measure(
            lambda: invoke_flagrand(
                distribution,
                timing_generator,
                timing_output,
                args.poisson_lambda,
            ),
            n,
            profile,
        )
        record.update(timing)
        add_throughput(record, output.element_size())
        record["status"] = "pass"
    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    return record


def run_torch_references(n: int, profile: Profile) -> list[dict[str, Any]]:
    cases: list[tuple[str, torch.dtype, Callable[[torch.Tensor], Any]]] = [
        ("uniform_f32", torch.float32, lambda out: out.uniform_()),
        ("normal_f32", torch.float32, lambda out: out.normal_(0.0, 1.0)),
        ("lognormal_f32", torch.float32, lambda out: out.log_normal_(0.0, 1.0)),
        ("uniform_f64", torch.float64, lambda out: out.uniform_()),
        ("normal_f64", torch.float64, lambda out: out.normal_(0.0, 1.0)),
        ("lognormal_f64", torch.float64, lambda out: out.log_normal_(0.0, 1.0)),
    ]
    records = [
        run_torch_case("torch_rng_reference", distribution, dtype, n, operation, profile)
        for distribution, dtype, operation in cases
    ]
    for dtype_name, dtype in (
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("float32", torch.float32),
        ("float64", torch.float64),
    ):
        records.append(
            run_torch_case(
                "zero_write_reference",
                f"zero_{dtype_name}",
                dtype,
                n,
                lambda out: out.zero_(),
                profile,
            )
        )
    return records


def run_torch_case(
    backend: str,
    distribution: str,
    dtype: torch.dtype,
    n: int,
    operation: Callable[[torch.Tensor], Any],
    profile: Profile,
) -> dict[str, Any]:
    record = base_record(
        backend=backend,
        generator="torch_default" if backend == "torch_rng_reference" else "not_applicable",
        distribution=distribution,
        n=n,
        dtype=dtype,
    )
    record["comparison_scope"] = (
        "platform_reference_not_algorithm_equivalent"
        if backend == "torch_rng_reference"
        else "write_reference_not_theoretical_bandwidth"
    )
    try:
        output = torch.empty(n, device="cuda", dtype=dtype)
        operation(output)
        torch.cuda.synchronize()
        if dtype.is_floating_point:
            finite = bool(torch.isfinite(output).all().item())
            metrics = tensor_metrics(output)
            validation = {"status": "pass" if finite else "fail", **metrics}
        else:
            validation = {"status": "pass", **tensor_metrics(output)}
        record["validation"] = validation
        if validation["status"] != "pass":
            record["status"] = "validation_fail"
            return record
        record.update(measure(lambda: operation(output), n, profile))
        add_throughput(record, output.element_size())
        record["status"] = "pass"
    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    return record


def make_generator(spec: GeneratorSpec, seed: int, dimensions: int) -> object:
    return curand.create_generator(
        spec.rng_type,
        seed=seed,
        offset=0,
        dimensions=dimensions,
    )


def invoke_flagrand(
    distribution: str,
    generator: object,
    output: torch.Tensor,
    poisson_lambda: float,
) -> None:
    if distribution == "raw32":
        curand.generate(generator, output)
    elif distribution == "raw64":
        curand.generate_long_long(generator, output)
    elif distribution == "uniform_f32":
        curand.generate_uniform(generator, output)
    elif distribution == "uniform_f64":
        curand.generate_uniform_double(generator, output)
    elif distribution == "normal_f32":
        curand.generate_normal(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "normal_f64":
        curand.generate_normal_double(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "lognormal_f32":
        curand.generate_lognormal(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "lognormal_f64":
        curand.generate_lognormal_double(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "poisson_u32":
        curand.generate_poisson(generator, output, lambda_val=poisson_lambda)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def validate_output(
    output: torch.Tensor,
    replay: torch.Tensor,
    distribution: str,
    *,
    poisson_lambda: float,
) -> dict[str, Any]:
    metrics = tensor_metrics(output)
    checks: dict[str, bool] = {
        "reproducible": bool(torch.equal(output, replay)),
        "nonempty": output.numel() > 0,
    }
    if distribution.startswith("raw"):
        checks["not_all_zero"] = bool(torch.any(output != 0).item())
    elif distribution.startswith("uniform"):
        checks["finite"] = bool(torch.isfinite(output).all().item())
        checks["range"] = metrics["min"] >= 0.0 and metrics["max"] <= 1.0
        checks["mean_sanity"] = abs(metrics["mean"] - 0.5) < 0.08
    elif distribution.startswith("normal"):
        checks["finite"] = bool(torch.isfinite(output).all().item())
        checks["mean_sanity"] = abs(metrics["mean"]) < 0.10
        checks["std_sanity"] = 0.80 < metrics["std"] < 1.20
    elif distribution.startswith("lognormal"):
        checks["finite"] = bool(torch.isfinite(output).all().item())
        checks["positive"] = metrics["min"] > 0.0
        checks["mean_sanity"] = 1.0 < metrics["mean"] < 2.5
    elif distribution == "poisson_u32":
        checks["nonnegative"] = metrics["min"] >= 0.0
        checks["mean_sanity"] = abs(metrics["mean"] - poisson_lambda) < max(
            0.4, poisson_lambda * 0.08
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        **metrics,
    }


def tensor_metrics(output: torch.Tensor) -> dict[str, float]:
    values = output.double()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
    }


def measure(fn: Callable[[], Any], n: int, profile: Profile) -> dict[str, Any]:
    calls = max(
        1,
        min(
            profile.max_calls_per_batch,
            math.ceil(profile.target_items_per_batch / n),
        ),
    )
    for _ in range(profile.warmup):
        fn()
    torch.cuda.synchronize()

    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(profile.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter()
        gpu_ms.append(float(start.elapsed_time(end)) / calls)
        wall_ms.append((wall_end - wall_start) * 1000.0 / calls)
    return {
        "calls_per_batch": calls,
        "warmup": profile.warmup,
        "repeats": profile.repeats,
        "median_gpu_ms": statistics.median(gpu_ms),
        "min_gpu_ms": min(gpu_ms),
        "max_gpu_ms": max(gpu_ms),
        "median_wall_sync_ms": statistics.median(wall_ms),
    }


def add_throughput(record: dict[str, Any], element_size: int) -> None:
    seconds = float(record["median_gpu_ms"]) / 1000.0
    n = int(record["n"])
    record["gsamples_per_s"] = n / seconds / 1e9 if seconds > 0 else None
    record["effective_output_gb_per_s"] = (
        n * element_size / seconds / 1e9 if seconds > 0 else None
    )


def base_record(
    *,
    backend: str,
    generator: str,
    distribution: str,
    n: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "generator": generator,
        "distribution": distribution,
        "n": n,
        "dtype": str(dtype).replace("torch.", ""),
        "status": "pending",
    }


def collect_environment(
    args: argparse.Namespace,
    profile: Profile,
    sizes: tuple[int, ...],
    compatibility: dict[str, Any],
    vendor_report: dict[str, Any],
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    target = driver.active.get_current_target()
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "profile": args.profile,
        "profile_config": {
            "sizes": list(sizes),
            "warmup": profile.warmup,
            "repeats": profile.repeats,
            "target_items_per_batch": profile.target_items_per_batch,
            "max_calls_per_batch": profile.max_calls_per_batch,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version_field": torch.version.cuda,
        "triton_version": triton.__version__,
        "triton_target": repr(target),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "major": props.major,
            "minor": props.minor,
        },
        "compatibility": compatibility,
        "curand": {
            "status": "unavailable",
            "ctypes_find_library": ctypes.util.find_library("curand"),
            "reason": "No libcurand-compatible shared library is present in the MXMACA container.",
        },
        "comparison_policy": {
            "torch_rng_reference": "same-device platform reference; not algorithm-equivalent",
            "zero_write_reference": "measured write reference; not theoretical memory bandwidth",
            "h20_results": "cross-device historical context only; no formal same-device speedup",
        },
        "git_commit": git_commit(),
        "flagrand_vendor": vendor_report,
    }


def git_commit() -> str | None:
    from_env = os.environ.get("CURAND_CONTRACT_GIT_SHA")
    if from_env:
        return from_env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def summarize(records: list[dict[str, Any]], environment: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(str(record.get("status")) for record in records)
    errors = [
        {
            "backend": record.get("backend"),
            "generator": record.get("generator"),
            "distribution": record.get("distribution"),
            "n": record.get("n"),
            "error_type": record.get("error_type"),
            "error": record.get("error"),
        }
        for record in records
        if record.get("status") in {"error", "validation_fail"}
    ]
    return {
        "run_health": "ok" if not errors else "needs_attention",
        "status_counts": dict(counts),
        "record_count": len(records),
        "curand_status": environment["curand"],
        "compatibility": environment["compatibility"],
        "errors": errors,
    }


def print_record(record: dict[str, Any]) -> None:
    prefix = (
        f"[{record.get('status')}] {record.get('backend')} "
        f"{record.get('generator')} {record.get('distribution')} n={record.get('n')}"
    )
    if record.get("status") == "pass" and record.get("median_gpu_ms") is not None:
        print(
            f"{prefix} gpu_ms={record['median_gpu_ms']:.6f} "
            f"GSample/s={record['gsamples_per_s']:.3f} "
            f"GB/s={record['effective_output_gb_per_s']:.3f}"
        )
    elif record.get("status") == "pass":
        print(prefix)
    else:
        print(f"{prefix} error={record.get('error') or record.get('validation')}")


def make_run_dir(base: Path, profile: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{stamp}_{profile}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({key for record in records for key in record if key not in {"validation", "traceback_tail", "parameters"}})
    fields.extend(["validation_json", "parameters_json", "traceback_tail_json"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: value for key, value in record.items() if key in fields}
            row["validation_json"] = json.dumps(record.get("validation"), ensure_ascii=False)
            row["parameters_json"] = json.dumps(record.get("parameters"), ensure_ascii=False)
            row["traceback_tail_json"] = json.dumps(record.get("traceback_tail"), ensure_ascii=False)
            writer.writerow(row)


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    lines = [
        "# MetaX C550 FlagRand Matrix",
        "",
        f"- run health: `{summary['run_health']}`",
        f"- device: `{environment['device']['name']}`",
        f"- Triton target: `{environment['triton_target']}`",
        f"- Torch: `{environment['torch_version']}`",
        f"- Triton: `{environment['triton_version']}`",
        f"- cuRAND: `unavailable`",
        f"- compatibility patches: `{environment['compatibility']['applied']}`",
        f"- status counts: `{summary['status_counts']}`",
        "",
        "cuRAND is not installed and is not synthesized from NVIDIA binaries. Torch RNG rows are same-device platform references, not algorithm-equivalent baselines. Zero-write rows are measured references, not theoretical bandwidth.",
        "",
        "## Largest-size passing results",
        "",
        "| backend | generator | distribution | n | GPU ms | GSample/s | effective GB/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "pass" or record.get("median_gpu_ms") is None:
            continue
        key = (str(record["backend"]), str(record["generator"]), str(record["distribution"]))
        if key not in groups or int(record["n"]) > int(groups[key]["n"]):
            groups[key] = record
    for key in sorted(groups):
        record = groups[key]
        lines.append(
            f"| {record['backend']} | {record['generator']} | {record['distribution']} | "
            f"{record['n']} | {record['median_gpu_ms']:.6f} | "
            f"{record['gsamples_per_s']:.3f} | {record['effective_output_gb_per_s']:.3f} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Errors and validation failures", ""])
        for error in summary["errors"]:
            lines.append(
                f"- `{error['backend']} / {error['generator']} / {error['distribution']} / n={error['n']}`: "
                f"{error.get('error_type')}: {error.get('error')}"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `environment.json`",
            "- `results.jsonl`",
            "- `results.csv`",
            "- `summary.json`",
            "- `REPORT.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise SystemExit("--sizes must contain positive integers")
    return values


def parse_name_set(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


if __name__ == "__main__":
    raise SystemExit(main())
