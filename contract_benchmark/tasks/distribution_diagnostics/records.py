from __future__ import annotations

from typing import Any, Callable

import torch

from contract_benchmark.adapters import flagrand_generate_by_distribution, flagrand_generate_raw, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import add_audit_flag, error_record, timed_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import dtype_for_distribution, validate_after, validate_distribution
from contract_benchmark.tasks.distribution_diagnostics.cases import (
    GENERATOR,
    DistributionCase,
    comparison_key,
    lambda_value,
    parameters,
)
from contract_benchmark.tasks.distribution_diagnostics.transforms import launch_transform, prepare_raw_and_output
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_raw_tensor

RecordRunner = Callable[[BenchmarkContext, TaskSpec, DistributionCase, int, str], dict[str, Any]]


def case_records(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component, runner in (
        ("raw_only", raw_only_record),
        ("transform_only", transform_only_record),
        ("raw_plus_transform", raw_plus_transform_record),
        ("public_api", public_api_record),
    ):
        try:
            rows.append(runner(ctx, spec, case, n, component))
        except BaseException as exc:
            rows.append(
                diagnostic_record(
                    error_record(ctx, spec, f"flagrand_diag_{component}", exc, generator=GENERATOR, distribution=case.distribution, n=n, parameters=parameters(case, component)),
                    component,
                )
            )
    return rows


def raw_only_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    raw = torch.empty(n, device=ctx.device, dtype=torch.int32)
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)
    run_once = lambda: flagrand_generate_raw(raw, gen)
    validation = validate_after(run_once, lambda: validate_raw_tensor(raw, dtype=torch.int32, n=n))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return diagnostic_record(
        timed_record(
            ctx,
            spec,
            "flagrand_diag_raw_only",
            "flagrand_public_raw_kernel",
            GENERATOR,
            case.distribution,
            n,
            timing,
            validation,
            parameters=parameters(case, component),
            comparison_key=comparison_key(spec, case, n, component),
            is_baseline=False,
            temporary_bytes=0,
            output_bytes=n * raw.element_size(),
        ),
        component,
        output_dtype=str(raw.dtype),
    )


def transform_only_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    raw, out = prepare_raw_and_output(ctx, case, n)
    run_once = lambda: launch_transform(case, raw, out, n)
    validation = validate_after(run_once, lambda: validate_distribution(out, case.distribution, n, lambda_val=lambda_value(case)))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return timed_diagnostic(ctx, spec, case, n, component, out, timing, validation, "flagrand_diag_transform_only", "flagrand_internal_transform_kernel", 0)


def raw_plus_transform_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    raw = torch.empty(n, device=ctx.device, dtype=torch.int32)
    out = torch.empty(n, device=ctx.device, dtype=dtype_for_distribution(case.distribution))
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)

    def run_once() -> torch.Tensor:
        flagrand_generate_raw(raw, gen)
        return launch_transform(case, raw, out, n)

    validation = validate_after(run_once, lambda: validate_distribution(out, case.distribution, n, lambda_val=lambda_value(case)))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return timed_diagnostic(
        ctx,
        spec,
        case,
        n,
        component,
        out,
        timing,
        validation,
        "flagrand_diag_raw_plus_transform",
        "flagrand_public_raw_plus_internal_transform",
        n * raw.element_size(),
    )


def public_api_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    out = torch.empty(n, device=ctx.device, dtype=dtype_for_distribution(case.distribution))
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)
    run_once = lambda: flagrand_generate_by_distribution(gen, out, case.distribution, lambda_val=lambda_value(case))
    validation = validate_after(run_once, lambda: validate_distribution(out, case.distribution, n, lambda_val=lambda_value(case)))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return timed_diagnostic(ctx, spec, case, n, component, out, timing, validation, "flagrand_diag_public_api", "flagrand_public_distribution_api", "internal_raw_allocation")


def timed_diagnostic(
    ctx: BenchmarkContext,
    spec: TaskSpec,
    case: DistributionCase,
    n: int,
    component: str,
    out: torch.Tensor,
    timing: Any,
    validation: dict[str, Any],
    backend: str,
    api_surface: str,
    temporary_bytes: int | str,
) -> dict[str, Any]:
    return diagnostic_record(
        timed_record(
            ctx,
            spec,
            backend,
            api_surface,
            GENERATOR,
            case.distribution,
            n,
            timing,
            validation,
            parameters=parameters(case, component),
            comparison_key=comparison_key(spec, case, n, component),
            is_baseline=False,
            temporary_bytes=temporary_bytes,
            output_bytes=n * out.element_size(),
        ),
        component,
        output_dtype=str(out.dtype),
    )


def diagnostic_record(record: dict[str, Any], component: str, *, output_dtype: str | None = None) -> dict[str, Any]:
    record["result_role"] = "diagnostic"
    record["diagnostic_component"] = component
    if output_dtype:
        record["diagnostic_output_dtype"] = output_dtype
        record["dtype"] = output_dtype.replace("torch.", "")
    record["formal_result"] = False
    add_audit_flag(record, "diagnostic_only")
    limitations = record.setdefault("known_limitations", [])
    note = "Diagnostic timing only; excluded from formal speedup claims."
    if note not in limitations:
        limitations.append(note)
    return record
