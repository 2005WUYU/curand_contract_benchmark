from __future__ import annotations

from typing import Any

import torch

from contract_benchmark.adapters import GENERATOR_INFOS, make_curand_generator
from contract_benchmark.curand_ctypes import CurandError
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import error_record, timed_record, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, measure_curand_and_flagrand_distribution, measure_curand_and_flagrand_raw, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_raw_tensor


def run_bulk_raw32(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.raw_generators:
        info = GENERATOR_INFOS.get(generator)
        if info is None or not info.supports_raw32:
            continue
        for n0 in ctx.profile.sizes:
            n = adjust_n(n0, generator, "raw32")
            records.extend(measure_curand_and_flagrand_raw(ctx, spec, generator, n, "raw32", torch.int32))
    return records


def run_bulk_raw64(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ("sobol64", "scrambled_sobol64"):
        if generator not in ctx.profile.qrng_generators and generator != "sobol64":
            continue
        for n in ctx.profile.sizes:
            records.extend(measure_curand_and_flagrand_raw(ctx, spec, generator, n, "raw64", torch.int64))
    return records


def run_bulk_distribution(ctx: BenchmarkContext, spec: TaskSpec, distribution: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.dist_generators:
        for n0 in ctx.profile.sizes:
            n = adjust_n(n0, generator, distribution)
            records.extend(measure_curand_and_flagrand_distribution(ctx, spec, generator, n, distribution))
    return records


def run_poisson(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    for lambda_val in ctx.profile.poisson_lambdas:
        for n0 in ctx.profile.sizes:
            n = adjust_n(n0, generator, "poisson_u32")
            records.extend(
                measure_curand_and_flagrand_distribution(
                    ctx,
                    spec,
                    generator,
                    n,
                    "poisson_u32",
                    parameters={"lambda": lambda_val},
                    lambda_val=lambda_val,
                )
            )
    return records


def run_ordering_sweep(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    n = adjust_n(max(ctx.profile.sizes), generator, "raw32")
    for ordering in ("legacy", "default", "best", "dynamic", "seeded"):
        out = torch.empty(n, device=ctx.device, dtype=torch.int32)
        try:
            with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering=ordering) as gen:
                validation = validate_after(lambda: gen.generate_raw_u32(out), lambda: validate_raw_tensor(out, dtype=torch.int32, n=n))
                timing = collect_cuda_event_and_wall_us(lambda: gen.generate_raw_u32(out), warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(
                    ctx,
                    spec,
                    "curand_host_ordering",
                    "curand_host_api",
                    generator,
                    "raw32",
                    n,
                    timing,
                    validation,
                    parameters={"ordering": ordering},
                    ordering=ordering,
                    comparison_key=f"{spec.task_id}:{generator}:{n}",
                    is_baseline=ordering == "legacy",
                    baseline_id="curand_host_legacy",
                )
            )
        except CurandError as exc:
            records.append(
                unsupported_record(
                    ctx,
                    spec,
                    "curand_host_ordering",
                    f"ordering={ordering} unsupported for {generator}: cuRAND status {exc.status}",
                    generator=generator,
                    distribution="raw32",
                    n=n,
                    parameters={"ordering": ordering},
                    comparison_key=f"{spec.task_id}:{generator}:{n}",
                    baseline_id="curand_host_legacy",
                )
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, "curand_host_ordering", exc, generator=generator, distribution="raw32", n=n, parameters={"ordering": ordering}))
    return records
