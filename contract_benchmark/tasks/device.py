from __future__ import annotations

from typing import Any

import torch

from contract_benchmark import kernels
from contract_benchmark.adapters import flagrand_generate_by_distribution, flagrand_generate_raw, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import base_record, timed_record, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import (
    adjust_n,
    alternate_baseline_record,
    device_fused_rows,
    has_backend_baseline,
    has_legacy_device_baseline,
    legacy_device_mapping_parameters,
    validate_after,
)
from contract_benchmark.tasks.curanddx import curanddx_compile_support_record, run_curanddx_raw_output, run_curanddx_uniform_output
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_finite_output, validate_raw_tensor, validate_uniform, validation_pass


def run_device_raw_output(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    parameters = legacy_device_mapping_parameters(ctx)
    if ext is None or not hasattr(ext, "philox_raw_u32"):
        records.append(
            unsupported_record(
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
            comparison_key = f"{spec.task_id}:legacy_device:philox4x32_10:raw32:{n}"
            out = torch.empty(n, device=ctx.device, dtype=torch.int32)
            run_once = lambda: ext.philox_raw_u32(out, ctx.seed, ctx.offset)
            validation = validate_after(run_once, lambda: validate_raw_tensor(out, dtype=torch.int32, n=n))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(
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
                    comparison_key=comparison_key,
                    is_baseline=True,
                    baseline_id="curand_legacy_device_output",
                    temporary_bytes=0,
                    output_bytes=n * 4,
                )
            )
            out_flagrand = torch.empty(n, device=ctx.device, dtype=torch.int32)
            gen = make_flagrand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset)
            run_flagrand = lambda: flagrand_generate_raw(out_flagrand, gen)
            validation_flagrand = validate_after(run_flagrand, lambda: validate_raw_tensor(out_flagrand, dtype=torch.int32, n=n))
            timing_flagrand = collect_cuda_event_and_wall_us(run_flagrand, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(
                    ctx,
                    spec,
                    "flagrand_public_output",
                    "flagrand_public_api",
                    "philox4x32_10",
                    "raw32",
                    n,
                    timing_flagrand,
                    validation_flagrand,
                    parameters={"comparison_baseline": "curand_legacy_device_output"},
                    comparison_key=comparison_key,
                    is_baseline=False,
                    baseline_id="curand_legacy_device_output",
                    temporary_bytes=0,
                    output_bytes=n * 4,
                )
            )
    curanddx_rows = run_curanddx_raw_output(ctx, spec)
    records.extend(curanddx_rows)
    records.extend(
        _flagrand_output_rows_for_baseline(
            ctx,
            spec,
            curanddx_rows,
            distribution="raw32",
            dtype=torch.int32,
            baseline_id="curanddx",
        )
    )
    return records


def run_device_uniform_output(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    parameters = legacy_device_mapping_parameters(ctx)
    if ext is None or not hasattr(ext, "philox_uniform"):
        records.append(
            unsupported_record(
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
            comparison_key = f"{spec.task_id}:legacy_device:philox4x32_10:uniform_f32:{n}"
            out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            run_once = lambda: ext.philox_uniform(out, ctx.seed, ctx.offset)
            validation = validate_after(run_once, lambda: validate_uniform(out, n=n, low_open=True))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(
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
                    comparison_key=comparison_key,
                    is_baseline=True,
                    baseline_id="curand_legacy_device_output",
                    temporary_bytes=0,
                    output_bytes=n * 4,
                )
            )
            out_flagrand = torch.empty(n, device=ctx.device, dtype=torch.float32)
            gen = make_flagrand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset)
            run_flagrand = lambda: flagrand_generate_by_distribution(gen, out_flagrand, "uniform_f32")
            validation_flagrand = validate_after(run_flagrand, lambda: validate_uniform(out_flagrand, n=n))
            timing_flagrand = collect_cuda_event_and_wall_us(run_flagrand, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(
                    ctx,
                    spec,
                    "flagrand_public_output",
                    "flagrand_public_api",
                    "philox4x32_10",
                    "uniform_f32",
                    n,
                    timing_flagrand,
                    validation_flagrand,
                    parameters={"comparison_baseline": "curand_legacy_device_output"},
                    comparison_key=comparison_key,
                    is_baseline=False,
                    baseline_id="curand_legacy_device_output",
                    temporary_bytes=0,
                    output_bytes=n * 4,
                )
            )
    curanddx_rows = run_curanddx_uniform_output(ctx, spec)
    records.extend(curanddx_rows)
    records.extend(
        _flagrand_output_rows_for_baseline(
            ctx,
            spec,
            curanddx_rows,
            distribution="uniform_f32",
            dtype=torch.float32,
            baseline_id="curanddx",
        )
    )
    return records


def run_m3_device_fused_consume(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    alpha = 0.25
    for n0 in ctx.profile.sizes:
        n = adjust_n(n0, "philox4x32_10", "uniform_f32")
        parameters = {"alpha": alpha, "operation": "y=x+alpha*(u-0.5)"}
        comparison_key = f"{spec.task_id}:uniform_add_consume:{n}"
        device_rows = device_fused_rows(
            ctx,
            spec,
            n=n,
            distribution="uniform_add_consume",
            parameters=parameters,
            comparison_key=comparison_key,
        )
        records.extend(device_rows)
        baseline_targets: list[tuple[str, str]] = []
        if has_legacy_device_baseline(device_rows):
            baseline_targets.append((f"{comparison_key}:legacy_device", "curand_legacy_device_fused"))
        if has_backend_baseline(device_rows, "curanddx_fused"):
            baseline_targets.append((f"{comparison_key}:curanddx", "curanddx_fused"))
        if not baseline_targets:
            continue
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_fused = lambda: kernels.fused_philox_add_uniform(x, out, seed=ctx.seed, alpha=alpha)
        validation = validate_after(run_fused, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        first_key, first_baseline = baseline_targets[0]
        flagrand_record = timed_record(
            ctx,
            spec,
            "flagrand_fused_philox",
            "flagrand_benchmark_kernel",
            "philox4x32_10",
            "uniform_add_consume",
            n,
            timing,
            validation,
            parameters={"alpha": alpha, "operation": "y=x+alpha*(u-0.5)", "comparison_baseline": first_baseline},
            comparison_key=first_key,
            is_baseline=False,
            baseline_id=first_baseline,
            temporary_bytes=0,
            output_bytes=n * 4,
        )
        records.append(flagrand_record)
        for comparison_key_alt, baseline_id in baseline_targets[1:]:
            records.append(alternate_baseline_record(flagrand_record, comparison_key_alt, baseline_id))
    return records


def _flagrand_output_rows_for_baseline(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    baseline_rows: list[dict[str, Any]],
    *,
    distribution: str,
    dtype: torch.dtype,
    baseline_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        if baseline.get("backend") != baseline_id or not baseline.get("is_baseline"):
            continue
        n = int(baseline.get("N") or 0)
        comparison_key = baseline.get("comparison_key")
        if n <= 0 or not comparison_key:
            continue
        out = torch.empty(n, device=ctx.device, dtype=dtype)
        gen = make_flagrand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset)
        if distribution == "raw32":
            run_flagrand = lambda: flagrand_generate_raw(out, gen)
            validation = validate_after(run_flagrand, lambda: validate_raw_tensor(out, dtype=dtype, n=n))
        else:
            run_flagrand = lambda: flagrand_generate_by_distribution(gen, out, distribution)
            validation = validate_after(run_flagrand, lambda: validate_uniform(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_flagrand, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(
            timed_record(
                ctx,
                spec,
                "flagrand_public_output",
                "flagrand_public_api",
                "philox4x32_10",
                distribution,
                n,
                timing,
                validation,
                parameters={"comparison_baseline": baseline_id},
                comparison_key=str(comparison_key),
                is_baseline=False,
                baseline_id=baseline_id,
                temporary_bytes=0,
                output_bytes=n * out.element_size(),
            )
        )
    return records


def run_e1_compile_support_matrix(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
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
            unsupported_record(
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
        record = base_record(
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

    records.append(curanddx_compile_support_record(ctx, spec))
    return records
