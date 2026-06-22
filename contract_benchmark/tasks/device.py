from __future__ import annotations

from typing import Any

import torch

from contract_benchmark import kernels
from contract_benchmark.adapters import capability_matrix, flagrand_generate_by_distribution, flagrand_generate_raw, make_flagrand_generator
from contract_benchmark.curanddx_support import curanddx_status
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import base_record, error_record, timed_record, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, curanddx_unsupported_record, legacy_device_mapping_parameters, merge_validations, run_legacy_device_fused_extension, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import unsupported, validate_finite_output, validate_raw_tensor, validate_uniform, validation_pass


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
    records.append(curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="raw32"))
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
    records.append(curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="uniform_f32"))
    return records


def run_m3_device_fused_consume(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    from contract_benchmark.optional_device_api import find_built_curand_device_extension

    records: list[dict[str, Any]] = []
    ext, ext_reason = find_built_curand_device_extension()
    alpha = 0.25
    if ext is None:
        records.append(
            unsupported_record(
                ctx,
                spec,
                "curand_legacy_device_fused",
                ext_reason or "legacy Device API extension unavailable",
                generator="philox4x32_10",
                distribution="uniform_add_consume",
                parameters={"alpha": alpha, **legacy_device_mapping_parameters(ctx)},
                baseline_id="curand_legacy_device_fused",
            )
        )
        records.append(curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="uniform_add_consume"))
        return records

    for n0 in ctx.profile.sizes:
        n = adjust_n(n0, "philox4x32_10", "uniform_f32")
        parameters = {"alpha": alpha, "operation": "y=x+alpha*(u-0.5)", **legacy_device_mapping_parameters(ctx)}
        comparison_key = f"{spec.task_id}:uniform_add_consume:{n}:legacy_device"
        try:
            legacy_rows = run_legacy_device_fused_extension(
                ctx,
                spec,
                ext,
                n=n,
                distribution="uniform_add_consume",
                parameters=parameters,
                comparison_key=comparison_key,
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, "curand_legacy_device_fused", exc, generator="philox4x32_10", distribution="uniform_add_consume", n=n, parameters=parameters))
            records.append(curanddx_unsupported_record(ctx, spec, generator="philox4x32_10", distribution="uniform_add_consume", n=n, parameters={"alpha": alpha, "operation": "y=x+alpha*(u-0.5)"}))
            continue
        records.extend(legacy_rows)

        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_fused = lambda: kernels.fused_philox_add_uniform(x, out, seed=ctx.seed, alpha=alpha)
        validation = validate_after(run_fused, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(
            timed_record(
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
            curanddx_unsupported_record(
                ctx,
                spec,
                generator="philox4x32_10",
                distribution="uniform_add_consume",
                n=n,
                parameters={"alpha": alpha, "operation": "y=x+alpha*(u-0.5)"},
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

    dx_status = curanddx_status()
    dx_checks = {
        "headers_available": bool(dx_status.get("headers_available")),
        "benchmark_extension_available": bool(dx_status.get("extension_available")),
    }
    dx_record = base_record(
        ctx,
        spec,
        "curanddx",
        "compile_support_matrix",
        "philox4x32_10",
        "compile_support",
        0,
        validation=unsupported(str(dx_status.get("unsupported_reason")), dx_checks),
        parameters={
            "mathdx_root": dx_status.get("mathdx_root"),
            "header_paths": dx_status.get("header_paths") or [],
            "include_dirs_checked": dx_status.get("include_dirs_checked") or [],
            "unsupported_reason": dx_status.get("unsupported_reason"),
        },
    )
    dx_record.update({"formal_result": False, "audit_flags": ["unsupported_backend"]})
    records.append(dx_record)
    return records
