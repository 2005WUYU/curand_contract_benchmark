from __future__ import annotations

from typing import Any

import torch

from contract_benchmark.adapters import GENERATOR_INFOS, curand_generate_by_distribution, flagrand_generate_by_distribution, flagrand_generate_raw, make_curand_generator, make_flagrand_generator
from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.records import error_record, gate_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n, dtype_for_distribution, validate_distribution
from contract_benchmark.validation import tensors_equal, validate_raw_tensor, validation_pass


def run_g0(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for generator in ctx.profile.raw_generators + ctx.profile.qrng_generators:
        if generator not in GENERATOR_INFOS:
            continue
        info = GENERATOR_INFOS[generator]
        n = adjust_n(ctx.profile.gate_n, generator, "raw32" if info.supports_raw32 else "raw64")
        distribution = "raw64" if info.supports_raw64 else "raw32"
        dtype = torch.int64 if info.supports_raw64 else torch.int32
        if info.supports_curand_host:
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            try:
                dims = 1 if info.kind == "qrng" else None
                with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, dimensions=dims) as gen:
                    curand_generate_by_distribution(gen, out, distribution)
                    torch.cuda.synchronize()
                validation = validate_raw_tensor(out, dtype=dtype, n=n)
                records.append(gate_record(ctx, spec, "curand_host", generator, distribution, n, validation))
            except BaseException as exc:
                records.append(error_record(ctx, spec, "curand_host", exc, generator=generator, distribution=distribution, n=n))
        if info.supports_flagrand:
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            try:
                gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset, dimensions=1)
                flagrand_generate_raw(out, gen)
                torch.cuda.synchronize()
                validation = validate_raw_tensor(out, dtype=dtype, n=n)
                records.append(gate_record(ctx, spec, "flagrand_public", generator, distribution, n, validation))
            except BaseException as exc:
                records.append(error_record(ctx, spec, "flagrand_public", exc, generator=generator, distribution=distribution, n=n))
    return records


def run_g1(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generator = "philox4x32_10"
    n = adjust_n(ctx.profile.gate_n, generator, "normal_f32")
    for distribution in ("uniform_f32", "normal_f32", "lognormal_f32", "poisson_u32"):
        dtype = dtype_for_distribution(distribution)
        for backend in ("curand_host", "flagrand_public"):
            out = torch.empty(n, device=ctx.device, dtype=dtype)
            try:
                if backend == "curand_host":
                    with make_curand_generator(generator, seed=ctx.seed, offset=ctx.offset, ordering="legacy") as gen:
                        curand_generate_by_distribution(gen, out, distribution, lambda_val=10.0)
                else:
                    gen = make_flagrand_generator(generator, seed=ctx.seed, offset=ctx.offset)
                    flagrand_generate_by_distribution(gen, out, distribution, lambda_val=10.0)
                torch.cuda.synchronize()
                validation = validate_distribution(out, distribution, n, lambda_val=10.0)
                records.append(gate_record(ctx, spec, backend, generator, distribution, n, validation))
            except BaseException as exc:
                records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution=distribution, n=n))
    return records


def run_g2(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generators = [
        generator
        for generator in ctx.profile.raw_generators
        if GENERATOR_INFOS[generator].kind == "prng" and GENERATOR_INFOS[generator].supports_raw32
    ]
    for generator in generators:
        info = GENERATOR_INFOS[generator]
        n = adjust_n(ctx.profile.gate_n, generator, "raw32")
        for backend in ("curand_host", "flagrand_public"):
            if backend == "curand_host" and not info.supports_curand_host:
                continue
            if backend == "flagrand_public" and not info.supports_flagrand:
                continue
            try:
                a = torch.empty(n, device=ctx.device, dtype=torch.int32)
                b = torch.empty_like(a)
                c = torch.empty_like(a)
                first = torch.empty_like(a)
                second = torch.empty_like(a)
                full = torch.empty(n * 2, device=ctx.device, dtype=torch.int32)
                d = torch.empty_like(a) if info.supports_offset else None
                if backend == "curand_host":
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(a)
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(b)
                    with make_curand_generator(generator, seed=ctx.seed + 1, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(c)
                    if d is not None:
                        with make_curand_generator(generator, seed=ctx.seed, offset=n, ordering="legacy") as gen:
                            gen.generate_raw_u32(d)
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(first)
                        gen.generate_raw_u32(second)
                    with make_curand_generator(generator, seed=ctx.seed, offset=0, ordering="legacy") as gen:
                        gen.generate_raw_u32(full)
                else:
                    gen_a = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    gen_b = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    gen_c = make_flagrand_generator(generator, seed=ctx.seed + 1, offset=0)
                    gen_split = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    gen_full = make_flagrand_generator(generator, seed=ctx.seed, offset=0)
                    flagrand_generate_raw(a, gen_a)
                    flagrand_generate_raw(b, gen_b)
                    flagrand_generate_raw(c, gen_c)
                    if d is not None:
                        gen_d = make_flagrand_generator(generator, seed=ctx.seed, offset=n)
                        flagrand_generate_raw(d, gen_d)
                    flagrand_generate_raw(first, gen_split)
                    flagrand_generate_raw(second, gen_split)
                    flagrand_generate_raw(full, gen_full)
                torch.cuda.synchronize()
                checks: dict[str, Any] = {
                    "same_seed_same_output": tensors_equal(a, b),
                    "changed_seed_changes_output": not tensors_equal(a, c),
                    "same_generator_second_call_advances": not tensors_equal(a, second),
                    "split_first_matches_full_prefix": tensors_equal(first, full[:n]),
                    "split_second_matches_full_suffix": tensors_equal(second, full[n:]),
                    "offset_check_applicable": "yes" if info.supports_offset else "not_supported_by_generator",
                }
                if d is not None:
                    checks["changed_offset_changes_output"] = not tensors_equal(a, d)
                validation = validation_pass(checks)
                records.append(gate_record(ctx, spec, backend, generator, "raw32", n, validation))
            except BaseException as exc:
                records.append(error_record(ctx, spec, backend, exc, generator=generator, distribution="raw32", n=n))
    return records


def run_g3(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records = []
    for n in ctx.profile.sizes:
        n = adjust_n(n, "philox4x32_10", "uniform_f32")
        counters = (n + 3) // 4
        checks = {
            "n_multiple_of_4": n % 4 == 0,
            "counter_count_positive": counters > 0,
            "counter_count": counters,
            "reserved_counter_increment": counters,
            "max_lane_index": n - 1,
        }
        records.append(gate_record(ctx, spec, "flagrand_fused_philox", "philox4x32_10", "philox_counter_budget", n, validation_pass(checks)))
    return records
