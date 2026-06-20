from __future__ import annotations

from typing import Any, Callable

import torch

from contract_benchmark.adapters import GENERATOR_INFOS, flagrand_generate_by_distribution, make_curand_generator, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import error_record, timed_record, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us, collect_wall_only_us
from contract_benchmark.validation import validate_uniform, validation_pass


def run_lifecycle(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"

    def curand_once() -> None:
        gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")
        gen.destroy()

    def flagrand_once() -> object:
        return make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)

    for backend, api_surface, run_once in (
        ("curand_host_lifecycle", "curand_host_api", curand_once),
        ("flagrand_lifecycle", "flagrand_public_api", flagrand_once),
    ):
        try:
            validation = validation_pass({"status_success": True})
            timing = collect_wall_only_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats, sync_cuda=True)
            records.append(
                timed_record(
                    ctx,
                    spec,
                    backend,
                    api_surface,
                    generator,
                    "lifecycle",
                    0,
                    timing,
                    validation,
                    comparison_key=f"{spec.task_id}:{generator}",
                    is_baseline=backend.startswith("curand"),
                    baseline_id="curand_host_lifecycle",
                )
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution="lifecycle", n=0))
    return records


def run_generate_seeds(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in [g for g in ctx.profile.raw_generators if GENERATOR_INFOS[g].kind == "prng"]:
        try:
            with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                validation = validate_after(gen.generate_seeds, lambda: validation_pass({"curand_status_success": True}))
                timing = collect_cuda_event_and_wall_us(gen.generate_seeds, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(
                    ctx,
                    spec,
                    "curand_host_generate_seeds",
                    "curand_host_api",
                    generator,
                    "generate_seeds",
                    0,
                    timing,
                    validation,
                    comparison_key=f"{spec.task_id}:{generator}",
                    is_baseline=True,
                    baseline_id="curand_host_generate_seeds",
                )
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, "curand_host_generate_seeds", exc, generator=generator, distribution="generate_seeds", n=0))
    records.append(unsupported_record(ctx, spec, "flagrand_no_exact_generate_seeds", "FlagRand has no Host API equivalent to curandGenerateSeeds."))
    return records


def make_uniform_once(
    ctx: BenchmarkContext,
    backend: str,
    generator: str,
    out: torch.Tensor,
) -> tuple[Callable[[], object], Callable[[], None]]:
    if backend == "curand_host":
        gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")

        def cleanup() -> None:
            gen.destroy()

        return lambda: gen.generate_uniform_f32(out), cleanup

    gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
    return lambda: flagrand_generate_by_distribution(gen, out, "uniform_f32"), lambda: None


def run_first_vs_steady(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    n = adjust_n(ctx.profile.gate_n, generator, "uniform_f32")
    for backend in ("curand_host", "flagrand_public"):
        try:
            if backend == "curand_host":
                api_surface = "curand_host_api"
            else:
                api_surface = "flagrand_public_api"

            validation_out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            validation_run, validation_cleanup = make_uniform_once(ctx, backend, generator, validation_out)
            try:
                validation = validate_after(validation_run, lambda: validate_uniform(validation_out, n=n))
            finally:
                validation_cleanup()

            timings = []
            for phase, warmup_iters, repeats in (
                ("first", 0, 1),
                ("steady", ctx.profile.warmup, ctx.profile.repeats),
            ):
                out = torch.empty(n, device=ctx.device, dtype=torch.float32)
                run_once, cleanup = make_uniform_once(ctx, backend, generator, out)
                try:
                    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=warmup_iters, repeats=repeats)
                finally:
                    cleanup()
                timings.append((phase, timing))

            for phase, timing in timings:
                records.append(
                    timed_record(
                        ctx,
                        spec,
                        f"{backend}_{phase}",
                        api_surface,
                        generator,
                        "uniform_f32",
                        n,
                        timing,
                        validation,
                        parameters={
                            "phase": phase,
                            "timing_semantics": "fresh_generator_after_separate_validation",
                            "first_warmup_iters": 0,
                        },
                        comparison_key=f"{spec.task_id}:{phase}:{generator}:{n}",
                        is_baseline=backend == "curand_host",
                        baseline_id=f"curand_host_{phase}",
                    )
                )
        except BaseException as exc:
            records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution="uniform_f32", n=n))
    return records
