from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.spec import TaskSpec

GENERATOR = "philox4x32_10"


@dataclass(frozen=True)
class DistributionCase:
    distribution: str
    lambda_val: float | None = None


def diagnostic_cases(ctx: BenchmarkContext) -> list[DistributionCase]:
    cases = [
        DistributionCase("uniform_f32"),
        DistributionCase("normal_f32"),
        DistributionCase("lognormal_f32"),
    ]
    cases.extend(DistributionCase("poisson_u32", lambda_val=value) for value in diagnostic_lambdas(ctx))
    return cases


def diagnostic_sizes(ctx: BenchmarkContext) -> list[int]:
    return [int(value) for value in pick_first_middle_last([int(value) for value in ctx.profile.sizes])]


def diagnostic_lambdas(ctx: BenchmarkContext) -> list[float]:
    values = [float(value) for value in ctx.profile.poisson_lambdas]
    if not values:
        return [0.1, 10.0, 1024.0]
    preferred = [0.1, 10.0, 1024.0]
    selected = [value for value in preferred if value in values]
    if len(selected) >= 3:
        return selected[:3]
    for value in pick_first_middle_last(values):
        if value not in selected:
            selected.append(value)
    return selected[:3]


def pick_first_middle_last(values: list[int] | list[float]) -> list[Any]:
    ordered = sorted(dict.fromkeys(values))
    if len(ordered) <= 3:
        return ordered
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def parameters(case: DistributionCase, component: str) -> dict[str, Any]:
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


def comparison_key(spec: TaskSpec, case: DistributionCase, n: int, component: str) -> str:
    lambda_part = "" if case.lambda_val is None else f":lambda={case.lambda_val}"
    return f"{spec.task_id}:{case.distribution}{lambda_part}:N={n}:{component}"


def lambda_value(case: DistributionCase) -> float:
    return float(case.lambda_val if case.lambda_val is not None else 10.0)
