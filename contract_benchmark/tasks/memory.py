from __future__ import annotations

from typing import Any

import torch

from contract_benchmark import kernels
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import timed_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_finite_output


def run_pure_write(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records = []
    for n in ctx.profile.sizes:
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_once = lambda: kernels.pure_write_f32(out, value=1.0)
        validation = validate_after(run_once, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(timed_record(ctx, spec, "triton_pure_write", "flagrand_benchmark_kernel", "none", "pure_write_f32", n, timing, validation, comparison_key=f"{spec.task_id}:{n}", is_baseline=True, output_bytes=n * 4))
    return records


def run_pregenerated_consume(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records = []
    alpha = 0.25
    for n in ctx.profile.sizes:
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        u = torch.rand(n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        run_once = lambda: kernels.consume_add_uniform(x, u, out, alpha=alpha)
        validation = validate_after(run_once, lambda: validate_finite_output(out, n=n))
        timing = collect_cuda_event_and_wall_us(run_once, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(timed_record(ctx, spec, "triton_pregenerated_consume", "flagrand_benchmark_kernel", "none", "consume_pregenerated_uniform", n, timing, validation, parameters={"alpha": alpha}, comparison_key=f"{spec.task_id}:{n}", is_baseline=True, output_bytes=n * 8))
    return records
