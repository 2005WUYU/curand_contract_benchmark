from __future__ import annotations

from typing import Any, Callable

import torch

from contract_benchmark.adapters import flagrand_generate_by_distribution, make_curand_generator, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import base_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.validation import validation_error, validation_pass


def run_e0(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
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
                    "expected_raised": "yes" if expected_raised else "no",
                    "raised_matches_expected": bool(outcome.get("raised")) == expected_raised,
                    "failure_not_aggregated_as_speedup": True,
                }
            )
            validation["observed_error"] = outcome
        except BaseException as exc:
            validation = validation_error(exc)
        records.append(
            base_record(ctx, spec, backend, "robustness", "various", "error_case", 0, validation=validation, parameters={"case": case_id})
        )
    return records


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

