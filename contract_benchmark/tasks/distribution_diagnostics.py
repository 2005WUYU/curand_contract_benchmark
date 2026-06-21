from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from contract_benchmark.adapters import flagrand_generate_by_distribution, flagrand_generate_raw, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import add_audit_flag, error_record, timed_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, dtype_for_distribution, validate_after, validate_distribution
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_raw_tensor

GENERATOR = "philox4x32_10"
BLOCK_SIZE = 512
NUM_WARPS = 4


@dataclass(frozen=True)
class DistributionCase:
    distribution: str
    lambda_val: float | None = None


def run_distribution_decomposition(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in _diagnostic_cases(ctx):
        for n0 in _diagnostic_sizes(ctx):
            n = adjust_n(n0, GENERATOR, case.distribution)
            records.extend(_case_records(ctx, spec, case, n))
    return records


def _case_records(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component, runner in (
        ("raw_only", _raw_only_record),
        ("transform_only", _transform_only_record),
        ("raw_plus_transform", _raw_plus_transform_record),
        ("public_api", _public_api_record),
    ):
        try:
            records.append(runner(ctx, spec, case, n, component))
        except BaseException as exc:
            records.append(_diagnostic_record(error_record(ctx, spec, f"flagrand_diag_{component}", exc, generator=GENERATOR, distribution=case.distribution, n=n, parameters=_parameters(case, component)), component))
    return records


def _raw_only_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    raw = torch.empty(n, device=ctx.device, dtype=torch.int32)
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)
    run_once = lambda: flagrand_generate_raw(raw, gen)
    validation = validate_after(run_once, lambda: validate_raw_tensor(raw, dtype=torch.int32, n=n))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return _diagnostic_record(
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
            parameters=_parameters(case, component),
            comparison_key=_comparison_key(spec, case, n, component),
            is_baseline=False,
            temporary_bytes=0,
            output_bytes=n * raw.element_size(),
        ),
        component,
        output_dtype=str(raw.dtype),
    )


def _transform_only_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    raw, out = _prepare_raw_and_output(ctx, case, n)
    run_once = lambda: _launch_transform(case, raw, out, n)
    validation = validate_after(run_once, lambda: validate_distribution(out, case.distribution, n, lambda_val=_lambda(case)))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return _diagnostic_record(
        timed_record(
            ctx,
            spec,
            "flagrand_diag_transform_only",
            "flagrand_internal_transform_kernel",
            GENERATOR,
            case.distribution,
            n,
            timing,
            validation,
            parameters=_parameters(case, component),
            comparison_key=_comparison_key(spec, case, n, component),
            is_baseline=False,
            temporary_bytes=0,
            output_bytes=n * out.element_size(),
        ),
        component,
        output_dtype=str(out.dtype),
    )


def _raw_plus_transform_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    raw = torch.empty(n, device=ctx.device, dtype=torch.int32)
    out = torch.empty(n, device=ctx.device, dtype=dtype_for_distribution(case.distribution))
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)

    def run_once() -> torch.Tensor:
        flagrand_generate_raw(raw, gen)
        return _launch_transform(case, raw, out, n)

    validation = validate_after(run_once, lambda: validate_distribution(out, case.distribution, n, lambda_val=_lambda(case)))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return _diagnostic_record(
        timed_record(
            ctx,
            spec,
            "flagrand_diag_raw_plus_transform",
            "flagrand_public_raw_plus_internal_transform",
            GENERATOR,
            case.distribution,
            n,
            timing,
            validation,
            parameters=_parameters(case, component),
            comparison_key=_comparison_key(spec, case, n, component),
            is_baseline=False,
            temporary_bytes=n * raw.element_size(),
            output_bytes=n * out.element_size(),
        ),
        component,
        output_dtype=str(out.dtype),
    )


def _public_api_record(ctx: BenchmarkContext, spec: TaskSpec, case: DistributionCase, n: int, component: str) -> dict[str, Any]:
    out = torch.empty(n, device=ctx.device, dtype=dtype_for_distribution(case.distribution))
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)
    run_once = lambda: flagrand_generate_by_distribution(gen, out, case.distribution, lambda_val=_lambda(case))
    validation = validate_after(run_once, lambda: validate_distribution(out, case.distribution, n, lambda_val=_lambda(case)))
    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
    return _diagnostic_record(
        timed_record(
            ctx,
            spec,
            "flagrand_diag_public_api",
            "flagrand_public_distribution_api",
            GENERATOR,
            case.distribution,
            n,
            timing,
            validation,
            parameters=_parameters(case, component),
            comparison_key=_comparison_key(spec, case, n, component),
            is_baseline=False,
            temporary_bytes="internal_raw_allocation",
            output_bytes=n * out.element_size(),
        ),
        component,
        output_dtype=str(out.dtype),
    )


def _prepare_raw_and_output(ctx: BenchmarkContext, case: DistributionCase, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.empty(n, device=ctx.device, dtype=torch.int32)
    out = torch.empty(n, device=ctx.device, dtype=dtype_for_distribution(case.distribution))
    gen = make_flagrand_generator(GENERATOR, seed=ctx.seed, offset=ctx.offset)
    flagrand_generate_raw(raw, gen)
    torch.cuda.synchronize()
    return raw, out


def _launch_transform(case: DistributionCase, raw: torch.Tensor, out: torch.Tensor, n: int) -> torch.Tensor:
    kernels = _transform_kernels()
    triton = kernels["triton"]

    if case.distribution == "uniform_f32":
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kernels["uniform32"][grid](out.view(-1), raw.view(-1), n, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
    elif case.distribution == "normal_f32":
        n_pairs = n // 2
        grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
        kernels["normal32"][grid](out.view(-1), raw.view(-1), n_pairs, 0.0, 1.0, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
    elif case.distribution == "lognormal_f32":
        n_pairs = n // 2
        grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
        kernels["lognormal32"][grid](out.view(-1), raw.view(-1), n_pairs, 0.0, 1.0, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
    elif case.distribution == "poisson_u32":
        lambda_val = _lambda(case)
        if lambda_val < 30.0:
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            kernels["poisson_small32"][grid](out.view(-1), raw.view(-1), n, lambda_val, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
        else:
            n_pairs = n // 2
            grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
            kernels["poisson_large32"][grid](out.view(-1), raw.view(-1), n_pairs, lambda_val, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
    else:
        raise ValueError(f"Unsupported diagnostic distribution: {case.distribution}")
    return out


@lru_cache(maxsize=1)
def _transform_kernels() -> dict[str, Any]:
    import triton
    from flagrand.fused.lognormal import _lognormal_transform_kernel_32
    from flagrand.fused.normal import _normal_transform_kernel_32
    from flagrand.fused.poisson import _poisson_transform_kernel_large_32, _poisson_transform_kernel_small_32
    from flagrand.fused.uniform import _uniform_transform_kernel_32

    return {
        "triton": triton,
        "uniform32": _uniform_transform_kernel_32,
        "normal32": _normal_transform_kernel_32,
        "lognormal32": _lognormal_transform_kernel_32,
        "poisson_small32": _poisson_transform_kernel_small_32,
        "poisson_large32": _poisson_transform_kernel_large_32,
    }


def _diagnostic_record(record: dict[str, Any], component: str, *, output_dtype: str | None = None) -> dict[str, Any]:
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


def _diagnostic_cases(ctx: BenchmarkContext) -> list[DistributionCase]:
    cases = [
        DistributionCase("uniform_f32"),
        DistributionCase("normal_f32"),
        DistributionCase("lognormal_f32"),
    ]
    cases.extend(DistributionCase("poisson_u32", lambda_val=value) for value in _diagnostic_lambdas(ctx))
    return cases


def _diagnostic_sizes(ctx: BenchmarkContext) -> list[int]:
    values = [int(value) for value in ctx.profile.sizes]
    return _pick_first_middle_last(values)


def _diagnostic_lambdas(ctx: BenchmarkContext) -> list[float]:
    values = [float(value) for value in ctx.profile.poisson_lambdas]
    if not values:
        return [0.1, 10.0, 1024.0]
    preferred = [0.1, 10.0, 1024.0]
    selected = [value for value in preferred if value in values]
    if len(selected) >= 3:
        return selected[:3]
    for value in _pick_first_middle_last(values):
        if value not in selected:
            selected.append(value)
    return selected[:3]


def _pick_first_middle_last(values: list[int] | list[float]) -> list[Any]:
    ordered = sorted(dict.fromkeys(values))
    if len(ordered) <= 3:
        return ordered
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def _parameters(case: DistributionCase, component: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "component": component,
        "diagnostic": True,
        "raw_source": "flagrand_public_raw32",
        "allocator_included": component == "public_api",
        "raw_generation_included": component in {"raw_only", "raw_plus_transform", "public_api"},
        "transform_included": component in {"transform_only", "raw_plus_transform", "public_api"},
    }
    if case.lambda_val is not None:
        params["lambda"] = case.lambda_val
    return params


def _comparison_key(spec: TaskSpec, case: DistributionCase, n: int, component: str) -> str:
    lambda_part = "" if case.lambda_val is None else f":lambda={case.lambda_val}"
    return f"{spec.task_id}:{case.distribution}{lambda_part}:N={n}:{component}"


def _lambda(case: DistributionCase) -> float:
    return float(case.lambda_val if case.lambda_val is not None else 10.0)
