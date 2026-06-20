from __future__ import annotations

from typing import Any

import torch

from contract_benchmark.adapters import GENERATOR_INFOS, flagrand_generate_by_distribution, make_curand_generator, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import error_record, timed_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import measure_curand_and_flagrand_raw, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_uniform


def run_q0(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.qrng_generators:
        info = GENERATOR_INFOS[generator]
        distribution = "raw64" if info.supports_raw64 else "raw32"
        dtype = torch.int64 if info.supports_raw64 else torch.int32
        for n in ctx.profile.sizes:
            records.extend(measure_curand_and_flagrand_raw(ctx, spec, generator, n, distribution, dtype, dimensions=1))
    return records


def run_q1(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dimensions = 4
    points = max(1024, ctx.profile.gate_n // dimensions)
    n = points * dimensions
    for generator in ("sobol32", "scrambled_sobol32"):
        for backend in ("curand_host", "flagrand_public"):
            out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            try:
                if backend == "curand_host":
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, dimensions=dimensions) as gen:
                        run_once = lambda: gen.generate_uniform_f32(out)
                        api_surface = "curand_host_api"
                        validation = validate_after(run_once, lambda: validate_uniform(out, n=n, low_open=True))
                        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
                else:
                    gen = make_flagrand_generator(generator, seed=ctx.seed, offset=0, dimensions=dimensions)
                    run_once = lambda: flagrand_generate_by_distribution(gen, out, "uniform_f32")
                    api_surface = "flagrand_public_api"
                    validation = validate_after(run_once, lambda: validate_uniform(out, n=n))
                    timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
                records.append(
                    timed_record(ctx, spec, backend, api_surface, generator, "sobol_unit_cube_f32", n, timing, validation, parameters={"points": points, "dimensions": dimensions, "layout": "dimension_major_flattened"}, comparison_key=f"{spec.task_id}:{generator}:{points}:{dimensions}", is_baseline=backend == "curand_host", baseline_id="curand_host_sobol")
                )
            except BaseException as exc:
                records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution="sobol_unit_cube_f32", n=n, parameters={"points": points, "dimensions": dimensions}))
    return records
