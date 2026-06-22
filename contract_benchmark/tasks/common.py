from __future__ import annotations

import copy
import json
from typing import Any, Callable

import torch

from contract_benchmark.adapters import (
    capability_matrix,
    curand_generate_by_distribution,
    flagrand_generate_by_distribution,
    make_curand_generator,
    make_flagrand_generator,
)
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import error_record, timed_record, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import (
    validate_finite_output,
    validate_lognormal,
    validate_mask,
    validate_normal,
    validate_poisson,
    validate_raw_tensor,
    validate_uniform,
    validation_error,
)


def validate_after(run_once: Callable[[], object], validate: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        run_once()
        torch.cuda.synchronize()
        return validate()
    except BaseException as exc:
        return validation_error(exc)


def validate_distribution(out: torch.Tensor, distribution: str, n: int, *, lambda_val: float = 10.0) -> dict[str, Any]:
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


def merge_validations(*validations: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    status = "pass"
    for idx, validation in enumerate(validations):
        if validation.get("status") != "pass":
            status = "fail"
        for key, value in validation.get("checks", {}).items():
            checks[f"{idx}_{key}"] = value
    return {"status": status, "checks": checks}


def dtype_for_distribution(distribution: str) -> torch.dtype:
    if distribution in {"raw32", "poisson_u32"}:
        return torch.int32
    if distribution == "raw64":
        return torch.int64
    if distribution.endswith("_f64"):
        return torch.float64
    return torch.float32


def adjust_n(n: int, generator: str, distribution: str) -> int:
    value = int(n)
    if generator == "philox4x32_10" or distribution in {"normal_f32", "lognormal_f32"}:
        value = max(4, value + (-value % 4))
    if distribution in {"normal_f32", "lognormal_f32"} and value % 2:
        value += 1
    return value


def measure_curand_and_flagrand_raw(
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
            validation = validate_after(run_once, lambda: validate_raw_tensor(out, dtype=dtype, n=n))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            if backend == "curand_host":
                gen.destroy()
            records.append(
                timed_record(ctx, spec, backend, api_surface, generator, distribution, n, timing, validation, comparison_key=f"{spec.task_id}:{generator}:{distribution}:{n}:{dimensions}", is_baseline=backend == "curand_host", baseline_id="curand_host_legacy", output_bytes=n * out.element_size())
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution=distribution, n=n))
    return records


def measure_curand_and_flagrand_distribution(
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
    dtype = dtype_for_distribution(distribution)
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
            validation = validate_after(run_once, lambda: validate_distribution(out, distribution, n, lambda_val=lambda_val))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            if backend == "curand_host":
                gen.destroy()
            records.append(
                timed_record(ctx, spec, backend, api_surface, generator, distribution, n, timing, validation, parameters=params, comparison_key=f"{spec.task_id}:{generator}:{distribution}:{n}:{json.dumps(params, sort_keys=True)}", is_baseline=backend == "curand_host", baseline_id="curand_host_legacy", output_bytes=n * out.element_size())
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution=distribution, n=n, parameters=params))
    return records


def legacy_device_mapping_parameters(ctx: BenchmarkContext) -> dict[str, Any]:
    return {
        "device_mapping": "curand_init(seed, sequence=linear_index, offset=absolute_offset)",
        "absolute_offset": ctx.offset,
        "host_order_exact_match": False,
    }


def curanddx_unsupported_record(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    *,
    generator: str,
    distribution: str,
    n: int = 0,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cap = capability_matrix()
    dx_status = cap.get("curanddx", {})
    row_parameters = dict(parameters or {})
    row_parameters.update(
        {
            "headers_available": dx_status.get("headers_available"),
            "header_paths": dx_status.get("header_paths") or [],
            "extension_available": dx_status.get("extension_available"),
        }
    )
    return unsupported_record(
        ctx,
        spec,
        "curanddx",
        cap.get("curanddx", {}).get("unsupported_reason") or "cuRANDDx headers/build integration are not configured in this local repository.",
        generator=generator,
        distribution=distribution,
        n=n,
        parameters=row_parameters,
        baseline_id="curanddx",
    )


def device_fused_rows(
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
            unsupported_record(
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
            rows.extend(run_legacy_device_fused_extension(ctx, spec, ext, n=n, distribution=distribution, parameters=parameters, comparison_key=f"{comparison_key}:legacy_device"))
        except BaseException as exc:
            rows.append(error_record(ctx, spec, "curand_legacy_device_fused", exc, generator="philox4x32_10", distribution=distribution, n=n, parameters=parameters))
    from contract_benchmark.tasks.curanddx import run_curanddx_fused_extension

    rows.extend(run_curanddx_fused_extension(ctx, spec, n=n, distribution=distribution, parameters=parameters, comparison_key=f"{comparison_key}:curanddx"))
    return rows


def has_legacy_device_baseline(rows: list[dict[str, Any]]) -> bool:
    return has_backend_baseline(rows, "curand_legacy_device_fused")


def has_backend_baseline(rows: list[dict[str, Any]], backend: str) -> bool:
    return any(row.get("backend") == backend and row.get("is_baseline") for row in rows)


def alternate_baseline_record(record: dict[str, Any], comparison_key: str, baseline_id: str) -> dict[str, Any]:
    alternate = copy.deepcopy(record)
    alternate["comparison_key"] = comparison_key
    alternate["baseline_id"] = baseline_id
    alternate["is_baseline"] = False
    parameters = dict(alternate.get("parameters") or {})
    parameters["comparison_baseline"] = baseline_id
    alternate["parameters"] = parameters
    alternate["speedup_gpu_vs_baseline"] = None
    alternate["speedup_wall_vs_baseline"] = None
    alternate["speedup_baseline_formal"] = False
    return alternate


def run_legacy_device_fused_extension(
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
        validation = validate_after(run_once, lambda: validate_mask(mask, n=n, p=p))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        output_bytes = n
    elif distribution == "uniform_add_consume":
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_once = lambda: ext.philox_add_uniform(x, out, ctx.seed, ctx.offset, alpha)
        validation = validate_after(run_once, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        output_bytes = n * 4
    elif distribution == "dropout":
        x = torch.ones(n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
        run_once = lambda: ext.philox_dropout(x, out, mask, ctx.seed, ctx.offset, p)
        validation = validate_after(run_once, lambda: merge_validations(validate_finite_output(out, n=n), validate_mask(mask, n=n, p=p)))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        output_bytes = n * 5
    else:
        return [unsupported_record(ctx, spec, "curand_legacy_device_fused", f"extension has no runner for distribution={distribution}", generator="philox4x32_10", distribution=distribution, n=n, parameters=parameters)]
    return [
        timed_record(
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
