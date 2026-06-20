from __future__ import annotations

from typing import Any

import torch

from contract_benchmark.adapters import flagrand_generate_by_distribution, make_curand_generator, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import error_record, timed_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.bulk import run_bulk_distribution
from contract_benchmark.tasks.common import adjust_n, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_uniform


def run_single_call_curve(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    return run_bulk_distribution(ctx, spec, "uniform_f32")


def run_many_small(ctx: BenchmarkContext, spec: TaskSpec, *, calls: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    chunk_n = adjust_n(ctx.profile.many_small_chunk_n, generator, "uniform_f32")
    total_n = chunk_n * calls
    for backend in ("curand_host", "flagrand_public"):
        chunks = [torch.empty(chunk_n, device=ctx.device, dtype=torch.float32) for _ in range(calls)]
        try:
            if backend == "curand_host":
                gen = make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy")
                api_surface = "curand_host_api"

                def run_once() -> None:
                    for chunk in chunks:
                        gen.generate_uniform_f32(chunk)

            else:
                gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
                api_surface = "flagrand_public_api"

                def run_once() -> None:
                    for chunk in chunks:
                        flagrand_generate_by_distribution(gen, chunk, "uniform_f32")

            validation = validate_after(run_once, lambda: validate_uniform(chunks[-1], n=chunk_n))
            timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            if backend == "curand_host":
                gen.destroy()
            records.append(
                timed_record(
                    ctx,
                    spec,
                    backend,
                    api_surface,
                    generator,
                    "uniform_f32",
                    total_n,
                    timing,
                    validation,
                    parameters={"calls": calls, "chunk_n": chunk_n, "total_n": total_n},
                    comparison_key=f"{spec.task_id}:{calls}:{chunk_n}",
                    is_baseline=backend == "curand_host",
                    baseline_id="curand_host_many_small",
                    output_bytes=total_n * 4,
                    item_count=total_n,
                )
            )
        except BaseException as exc:
            records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution="uniform_f32", n=total_n, parameters={"calls": calls, "chunk_n": chunk_n}))
    return records
