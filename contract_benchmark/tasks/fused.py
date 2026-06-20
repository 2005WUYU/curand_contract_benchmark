from __future__ import annotations

from typing import Any

import torch

from contract_benchmark import kernels
from contract_benchmark.adapters import make_curand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import timed_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, alternate_baseline_record, device_fused_rows, has_legacy_device_baseline, merge_validations, validate_after
from contract_benchmark.timing import collect_cuda_event_and_wall_us
from contract_benchmark.validation import validate_finite_output, validate_mask


def run_threshold(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in ctx.profile.fused_ps:
        for n0 in ctx.profile.sizes:
            n = adjust_n(n0, "philox4x32_10", "uniform_f32")
            comparison_key = f"{spec.task_id}:{p}:{n}"
            u = torch.empty(n, device=ctx.device, dtype=torch.float32)
            mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
            with make_curand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                run_base = lambda: (gen.generate_uniform_f32(u), kernels.threshold_from_uniform(u, mask, p=p))
                validation = validate_after(run_base, lambda: validate_mask(mask, n=n, p=p))
                timing = collect_cuda_event_and_wall_us(run_base, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(ctx, spec, "curand_host_uniform_plus_threshold", "curand_host_api+triton_consume", "philox4x32_10", "uniform_threshold", n, timing, validation, parameters={"p": p}, comparison_key=comparison_key, is_baseline=True, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=n * 4)
            )
            mask2 = torch.empty(n, device=ctx.device, dtype=torch.uint8)
            run_fused = lambda: kernels.fused_philox_threshold(mask2, seed=ctx.seed, p=p)
            validation2 = validate_after(run_fused, lambda: validate_mask(mask2, n=n, p=p))
            timing2 = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            flagrand_record = timed_record(
                ctx, spec, "flagrand_fused_philox", "flagrand_benchmark_kernel", "philox4x32_10", "uniform_threshold", n, timing2, validation2, parameters={"p": p}, comparison_key=comparison_key, is_baseline=False, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=0
            )
            records.append(flagrand_record)
            device_rows = device_fused_rows(ctx, spec, n=n, distribution="uniform_threshold", parameters={"p": p}, comparison_key=comparison_key)
            records.extend(device_rows)
            if has_legacy_device_baseline(device_rows):
                records.append(alternate_baseline_record(flagrand_record, f"{comparison_key}:legacy_device", "curand_legacy_device_fused"))
    return records


def run_add_uniform(ctx: BenchmarkContext, spec: TaskSpec, *, task_override: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_id = task_override or spec.task_id
    alpha = 0.25
    for n0 in ctx.profile.sizes:
        n = adjust_n(n0, "philox4x32_10", "uniform_f32")
        x = torch.linspace(0, 1, n, device=ctx.device, dtype=torch.float32)
        u = torch.empty(n, device=ctx.device, dtype=torch.float32)
        out = torch.empty(n, device=ctx.device, dtype=torch.float32)
        with make_curand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
            run_base = lambda: (gen.generate_uniform_f32(u), kernels.consume_add_uniform(x, u, out, alpha=alpha))
            validation = validate_after(run_base, lambda: validate_finite_output(out, n=n))
            timing = collect_cuda_event_and_wall_us(run_base, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        records.append(
            timed_record(ctx, spec, "curand_host_bulk_plus_consume", "curand_host_api+triton_consume", "philox4x32_10", "uniform_add_consume", n, timing, validation, parameters={"alpha": alpha}, comparison_key=f"{task_id}:{n}", is_baseline=True, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=n * 4)
        )
        out2 = torch.empty_like(out)
        run_fused = lambda: kernels.fused_philox_add_uniform(x, out2, seed=ctx.seed, alpha=alpha)
        validation2 = validate_after(run_fused, lambda: validate_finite_output(out2, n=n))
        timing2 = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
        rec = timed_record(ctx, spec, "flagrand_fused_philox", "flagrand_benchmark_kernel", "philox4x32_10", "uniform_add_consume", n, timing2, validation2, parameters={"alpha": alpha}, comparison_key=f"{task_id}:{n}", is_baseline=False, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=0)
        if task_override:
            rec["task_id"] = task_override
        records.append(rec)
        if task_override is None:
            device_rows = device_fused_rows(ctx, spec, n=n, distribution="uniform_add_consume", parameters={"alpha": alpha}, comparison_key=f"{task_id}:{n}")
            records.extend(device_rows)
            if has_legacy_device_baseline(device_rows):
                records.append(alternate_baseline_record(rec, f"{task_id}:{n}:legacy_device", "curand_legacy_device_fused"))
    return records


def run_dropout(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in ctx.profile.fused_ps:
        for n0 in ctx.profile.sizes:
            n = adjust_n(n0, "philox4x32_10", "uniform_f32")
            comparison_key = f"{spec.task_id}:{p}:{n}"
            x = torch.ones(n, device=ctx.device, dtype=torch.float32)
            u = torch.empty(n, device=ctx.device, dtype=torch.float32)
            out = torch.empty(n, device=ctx.device, dtype=torch.float32)
            mask = torch.empty(n, device=ctx.device, dtype=torch.uint8)
            with make_curand_generator("philox4x32_10", seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                run_base = lambda: (gen.generate_uniform_f32(u), kernels.dropout_from_uniform(x, u, out, mask, p=p))
                validation = validate_after(run_base, lambda: merge_validations(validate_finite_output(out, n=n), validate_mask(mask, n=n, p=p)))
                timing = collect_cuda_event_and_wall_us(run_base, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            records.append(
                timed_record(ctx, spec, "curand_host_uniform_plus_dropout", "curand_host_api+triton_consume", "philox4x32_10", "dropout", n, timing, validation, parameters={"p": p, "rule": "u<=p", "scaling": "inverted"}, comparison_key=comparison_key, is_baseline=True, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=n * 4)
            )
            out2 = torch.empty_like(out)
            mask2 = torch.empty_like(mask)
            run_fused = lambda: kernels.fused_philox_dropout(x, out2, mask2, seed=ctx.seed, p=p)
            validation2 = validate_after(run_fused, lambda: merge_validations(validate_finite_output(out2, n=n), validate_mask(mask2, n=n, p=p)))
            timing2 = collect_cuda_event_and_wall_us(run_fused, warmup_iters=ctx.profile.warmup, repeats=ctx.profile.repeats)
            flagrand_record = timed_record(
                ctx, spec, "flagrand_fused_philox", "flagrand_benchmark_kernel", "philox4x32_10", "dropout", n, timing2, validation2, parameters={"p": p, "rule": "u<=p", "scaling": "inverted"}, comparison_key=comparison_key, is_baseline=False, baseline_id="curand_host_bulk_plus_consume", temporary_bytes=0
            )
            records.append(flagrand_record)
            device_rows = device_fused_rows(ctx, spec, n=n, distribution="dropout", parameters={"p": p}, comparison_key=comparison_key)
            records.extend(device_rows)
            if has_legacy_device_baseline(device_rows):
                records.append(alternate_baseline_record(flagrand_record, f"{comparison_key}:legacy_device", "curand_legacy_device_fused"))
    return records
