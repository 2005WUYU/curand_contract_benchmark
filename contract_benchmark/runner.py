from __future__ import annotations

from typing import Any, Callable

from contract_benchmark.adapters import capability_matrix
from contract_benchmark.profiles import BenchmarkContext, BenchmarkProfile, PROFILES, collect_environment
from contract_benchmark.records import error_record, finalize_records, metadata_record, metadata_rows, unsupported_record
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.bulk import run_bulk_distribution, run_bulk_raw32, run_bulk_raw64, run_ordering_sweep, run_poisson
from contract_benchmark.tasks.device import run_device_raw_output, run_device_uniform_output, run_e1_compile_support_matrix, run_m3_device_fused_consume
from contract_benchmark.tasks.distribution_diagnostics import run_distribution_decomposition
from contract_benchmark.tasks.fused import run_add_uniform, run_dropout, run_threshold
from contract_benchmark.tasks.gates import run_g0, run_g1, run_g2, run_g3
from contract_benchmark.tasks.granularity import run_many_small, run_single_call_curve
from contract_benchmark.tasks.lifecycle import run_first_vs_steady, run_generate_seeds, run_lifecycle
from contract_benchmark.tasks.memory import run_pregenerated_consume, run_pure_write
from contract_benchmark.tasks.qrng import run_q0, run_q1
from contract_benchmark.tasks.robustness import run_e0

TaskRunner = Callable[[BenchmarkContext, TaskSpec], list[dict[str, Any]]]


def run_specs(ctx: BenchmarkContext, selected_specs: list[TaskSpec]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cap_matrix: dict[str, Any] | None = None
    runners = _task_runners()
    for spec in selected_specs:
        try:
            if spec.task_id == "C0_CAPABILITY_MATRIX":
                cap_matrix = capability_matrix()
                records.extend(metadata_rows(ctx, spec, cap_matrix))
            elif spec.task_id == "C1_VERSION_SYMBOL_SELFTEST":
                records.append(metadata_record(ctx, spec, "version_symbol_selftest", collect_environment(ctx.profile.name)))
            else:
                runner = runners.get(spec.task_id)
                if runner is None:
                    records.append(unsupported_record(ctx, spec, "runner", f"No runner implemented for {spec.task_id}"))
                else:
                    records.extend(runner(ctx, spec))
        except BaseException as exc:
            records.append(error_record(ctx, spec, "runner", exc))
    finalize_records(records)
    if cap_matrix is None:
        cap_matrix = capability_matrix()
    return records, cap_matrix


def _task_runners() -> dict[str, TaskRunner]:
    return {
        "G0_BASIC_CONTRACT": run_g0,
        "G1_DISTRIBUTION_ROUGH_CHECK": run_g1,
        "G2_REPRODUCIBILITY": run_g2,
        "G3_SEQUENCE_COUNTER_BUDGET": run_g3,
        "H0_RAW32_BULK": run_bulk_raw32,
        "H1_RAW64_SOBOL_BULK": run_bulk_raw64,
        "H2_UNIFORM_F32_BULK": lambda c, s: run_bulk_distribution(c, s, "uniform_f32"),
        "H3_NORMAL_F32_BULK": lambda c, s: run_bulk_distribution(c, s, "normal_f32"),
        "H4_LOGNORMAL_F32_BULK": lambda c, s: run_bulk_distribution(c, s, "lognormal_f32"),
        "H5_POISSON_LAMBDA_SWEEP": run_poisson,
        "D0_DISTRIBUTION_DECOMPOSITION": run_distribution_decomposition,
        "H6_ORDERING_SWEEP": run_ordering_sweep,
        "I1_GENERATOR_LIFECYCLE": run_lifecycle,
        "I2_CURAND_GENERATE_SEEDS": run_generate_seeds,
        "I3_FIRST_VS_STEADY": run_first_vs_steady,
        "A0_SINGLE_CALL_CURVE": run_single_call_curve,
        "A1_FIXED_TOTAL_MANY_SMALL": lambda c, s: run_many_small(c, s, calls=c.profile.many_small_calls),
        "A2_FIXED_CHUNK_CALLS_SWEEP": lambda c, s: [row for calls in [1, max(2, c.profile.many_small_calls // 4), c.profile.many_small_calls] for row in run_many_small(c, s, calls=calls)],
        "K0_DEVICE_RAW_OUTPUT": run_device_raw_output,
        "K1_DEVICE_UNIFORM_OUTPUT": run_device_uniform_output,
        "M3_DEVICE_DX_FUSED_CONSUME": run_m3_device_fused_consume,
        "E1_COMPILE_SUPPORT_MATRIX": run_e1_compile_support_matrix,
        "F0_THRESHOLD_BERNOULLI": run_threshold,
        "F1_ADD_UNIFORM": run_add_uniform,
        "F2_DROPOUT": run_dropout,
        "M0_PURE_WRITE": run_pure_write,
        "M1_PREGENERATED_CONSUME": run_pregenerated_consume,
        "M2_HOST_BULK_CONSUME": lambda c, s: run_add_uniform(c, s, task_override="M2_HOST_BULK_CONSUME"),
        "Q0_RAW_SOBOL": run_q0,
        "Q1_SOBOL_D_DIM_UNIT_CUBE": run_q1,
        "E0_HOST_STATUS_MATRIX": run_e0,
    }
