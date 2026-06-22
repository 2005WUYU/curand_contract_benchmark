from __future__ import annotations

from typing import Any

import torch

from contract_benchmark.optional_curanddx_api import REQUIRED_SYMBOLS
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import base_record, error_record, timed_record, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, curanddx_unsupported_record, merge_validations, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_finite_output, validate_mask, validate_raw_tensor, validate_uniform, validation_pass


def run_curanddx_raw_output(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    return _run_curanddx_output(ctx, spec, distribution="raw32", dtype=torch.int32, symbol="philox_raw_u32")


def run_curanddx_uniform_output(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    return _run_curanddx_output(ctx, spec, distribution="uniform_f32", dtype=torch.float32, symbol="philox_uniform")


def run_curanddx_fused_extension(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    *,
    n: int,
    distribution: str,
    parameters: dict[str, Any],
    comparison_key: str,
) -> list[dict[str, Any]]:
    from contract_benchmark.optional_curanddx_api import find_built_curanddx_extension

    ext, reason = find_built_curanddx_extension()
    if ext is None:
        return [
            unsupported_record(
                ctx,
                spec,
                "curanddx_fused",
                reason or "cuRANDDx extension unavailable",
                generator="philox4x32_10",
                distribution=distribution,
                n=n,
                parameters=_curanddx_parameters(ctx, parameters),
                comparison_key=comparison_key,
                baseline_id="curanddx_fused",
            )
        ]

    try:
        p = float(parameters.get("p", 0.5))
        alpha = float(parameters.get("alpha", 0.25))
        row_parameters = _curanddx_parameters(ctx, parameters)
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
            validation = validate_after(
                run_once,
                lambda: merge_validations(
                    validate_finite_output(out, n=n),
                    validate_mask(mask, n=n, p=p),
                ),
            )
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            output_bytes = n * 5
        else:
            return [
                unsupported_record(
                    ctx,
                    spec,
                    "curanddx_fused",
                    f"cuRANDDx extension has no runner for distribution={distribution}",
                    generator="philox4x32_10",
                    distribution=distribution,
                    n=n,
                    parameters=_curanddx_parameters(ctx, parameters),
                    comparison_key=comparison_key,
                    baseline_id="curanddx_fused",
                )
            ]
    except BaseException as exc:
        return [
            error_record(
                ctx,
                spec,
                "curanddx_fused",
                exc,
                generator="philox4x32_10",
                distribution=distribution,
                n=n,
                parameters=_curanddx_parameters(ctx, parameters),
            )
        ]

    return [
        timed_record(
            ctx,
            spec,
            "curanddx_fused",
            "curanddx_extension",
            "philox4x32_10",
            distribution,
            n,
            timing,
            validation,
            parameters=row_parameters,
            comparison_key=comparison_key,
            is_baseline=True,
            baseline_id="curanddx_fused",
            temporary_bytes=0,
            output_bytes=output_bytes,
        )
    ]


def curanddx_compile_support_record(ctx: BenchmarkContext, spec: TaskSpec) -> dict[str, Any]:
    from contract_benchmark.curanddx_support import curanddx_status

    status = curanddx_status()
    checks = {
        "headers_available": bool(status.get("headers_available")),
        "extension_available": bool(status.get("extension_available")),
    }
    checks.update({f"has_{symbol}": bool((status.get("extension_symbols") or {}).get(symbol)) for symbol in REQUIRED_SYMBOLS})
    if not status.get("available"):
        record = base_record(
            ctx,
            spec,
            "curanddx",
            "compile_support_matrix",
            "philox4x32_10",
            "compile_support",
            0,
            validation={
                "status": "unsupported",
                "unsupported_reason": str(status.get("unsupported_reason")),
                "checks": checks,
            },
            parameters=_status_parameters(status),
        )
        record.update({"formal_result": False, "audit_flags": ["unsupported_backend"]})
        return record

    validation = validation_pass(checks)
    record = base_record(
        ctx,
        spec,
        "curanddx",
        "compile_support_matrix",
        "philox4x32_10",
        "compile_support",
        0,
        validation=validation,
        parameters=_status_parameters(status),
    )
    record.update({"formal_result": validation.get("status") == "pass", "audit_flags": []})
    return record


def _run_curanddx_output(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    *,
    distribution: str,
    dtype: torch.dtype,
    symbol: str,
) -> list[dict[str, Any]]:
    from contract_benchmark.optional_curanddx_api import find_built_curanddx_extension

    ext, reason = find_built_curanddx_extension()
    if ext is None or not hasattr(ext, symbol):
        return [
            curanddx_unsupported_record(
                ctx,
                spec,
                generator="philox4x32_10",
                distribution=distribution,
                parameters={"missing_symbol": symbol, "loader_reason": reason},
            )
        ]

    rows: list[dict[str, Any]] = []
    for n0 in ctx.profile.sizes:
        n = adjust_n(n0, "philox4x32_10", distribution)
        try:
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            run_once = lambda: getattr(ext, symbol)(out, ctx.seed, ctx.offset)
            if distribution == "raw32":
                validation = validate_after(run_once, lambda: validate_raw_tensor(out, dtype=dtype, n=n))
                output_bytes = n * 4
            else:
                validation = validate_after(run_once, lambda: validate_uniform(out, n=n))
                output_bytes = n * out.element_size()
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            rows.append(
                timed_record(
                    ctx,
                    spec,
                    "curanddx",
                    "curanddx_extension",
                    "philox4x32_10",
                    distribution,
                    n,
                    timing,
                    validation,
                    parameters=_curanddx_parameters(ctx),
                    comparison_key=f"{spec.task_id}:curanddx:philox4x32_10:{distribution}:{n}",
                    is_baseline=True,
                    baseline_id="curanddx",
                    temporary_bytes=0,
                    output_bytes=output_bytes,
                )
            )
        except BaseException as exc:
            rows.append(
                error_record(
                    ctx,
                    spec,
                    "curanddx",
                    exc,
                    generator="philox4x32_10",
                    distribution=distribution,
                    n=n,
                    parameters=_curanddx_parameters(ctx),
                )
            )
    return rows


def _curanddx_parameters(ctx: BenchmarkContext, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    parameters = {
        "device_mapping": "cuRANDDx Philox4_32 generate4; subsequence=(offset/4+group)%65536, offset=(offset/4+group)/65536",
        "absolute_offset_elements": ctx.offset,
        "host_ordering_target": "CURAND_ORDERING_PSEUDO_LEGACY raw-bit order when offset is 4-aligned",
        "native_source": "native/curanddx_contract_ext.cu",
    }
    if extra:
        parameters.update(extra)
    return parameters


def _status_parameters(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "mathdx_root": status.get("mathdx_root"),
        "header_paths": status.get("header_paths") or [],
        "include_dirs_checked": status.get("include_dirs_checked") or [],
        "extension_available": status.get("extension_available"),
        "extension_build_dir": status.get("extension_build_dir"),
        "extension_symbols": status.get("extension_symbols") or {},
        "unsupported_reason": status.get("unsupported_reason"),
        "source": "native/curanddx_contract_ext.cu",
    }
