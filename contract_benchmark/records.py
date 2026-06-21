from __future__ import annotations

from typing import Any

from contract_benchmark.timing import audit_flags, formal_result_from_flags
from contract_benchmark.validation import unsupported, validation_error, validation_pass

PUBLIC_OUTPUT_BACKEND_GATE_ALIASES = {
    "flagrand_public_first": "flagrand_public",
    "flagrand_public_output": "flagrand_public",
    "flagrand_public_steady": "flagrand_public",
}

CURAND_HOST_OUTPUT_BACKEND_GATE_ALIASES = {
    "curand_host_bulk_plus_consume": "curand_host",
    "curand_host_first": "curand_host",
    "curand_host_ordering": "curand_host",
    "curand_host_steady": "curand_host",
    "curand_host_uniform_plus_dropout": "curand_host",
    "curand_host_uniform_plus_threshold": "curand_host",
}

DIAGNOSTIC_FLAGRAND_BACKENDS = {
    "flagrand_diag_public_api",
    "flagrand_diag_raw_only",
    "flagrand_diag_raw_plus_transform",
    "flagrand_diag_transform_only",
}

DIAGNOSTIC_DISTRIBUTION_COMPONENTS = {
    "public_api",
    "raw_plus_transform",
    "transform_only",
}

POISSON_NORMAL_APPROXIMATION_NOTE = (
    "FlagRand lambda >= 30 Poisson uses a normal approximation; use these rows "
    "as accepted approximate-Poisson performance, not strict cuRAND-equivalent Poisson."
)


def timed_record(
    ctx: Any,
    spec: Any,
    backend: str,
    api_surface: str,
    generator: str,
    distribution: str,
    n: int,
    timing: Any,
    validation: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
    comparison_key: str,
    is_baseline: bool,
    baseline_id: str | None = None,
    ordering: str | None = "legacy",
    temporary_bytes: int | str | None = "not_exposed",
    output_bytes: int | None = None,
    item_count: int | None = None,
) -> dict[str, Any]:
    flags = audit_flags(timing)
    record = base_record(
        ctx,
        spec,
        backend,
        api_surface,
        generator,
        distribution,
        n,
        validation=validation,
        parameters=parameters,
        ordering=ordering,
    )
    record.update(timing.to_record())
    record.update(
        {
            "comparison_key": comparison_key,
            "is_baseline": is_baseline,
            "baseline_id": baseline_id,
            "temporary_bytes": temporary_bytes,
            "audit_flags": flags,
            "formal_result": validation.get("status") == "pass" and formal_result_from_flags(flags),
        }
    )
    median = record.get("median_gpu_us")
    items = item_count if item_count is not None else n
    bytes_out = output_bytes if output_bytes is not None else _estimate_output_bytes(distribution, n)
    if median and median > 0 and items:
        record["items_per_second"] = items / (median * 1e-6)
    if median and median > 0 and bytes_out:
        record["bytes_per_second"] = bytes_out / (median * 1e-6)
    return record


def base_record(
    ctx: Any,
    spec: Any,
    backend: str,
    api_surface: str,
    generator: str,
    distribution: str,
    n: int,
    *,
    validation: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    ordering: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": spec.task_id,
        "claim_id": spec.claim_id,
        "family": spec.family,
        "comparison_level": spec.comparison_level,
        "observability_class": spec.observability_class,
        "backend": backend,
        "gate_backend": canonical_gate_backend(backend),
        "api_surface": api_surface,
        "generator": generator,
        "distribution": distribution,
        "dtype": _dtype_name(distribution),
        "ordering": ordering,
        "seed": ctx.seed,
        "offset": ctx.offset,
        "N": int(n),
        "parameters": parameters or {},
        "timing_boundary": spec.timing_boundary,
        "validation": validation,
        "validation_scope": "rough_gate",
        "what_it_can_say": spec.can_say,
        "what_it_cannot_say": spec.cannot_say,
        "known_limitations": [spec.cannot_say],
    }


def metadata_record(ctx: Any, spec: Any, backend: str, payload: dict[str, Any]) -> dict[str, Any]:
    record = base_record(ctx, spec, backend, "metadata", "not_applicable", "metadata", 0, validation=validation_pass({"metadata_recorded": True}))
    record["payload"] = payload
    record["formal_result"] = True
    return record


def metadata_rows(ctx: Any, spec: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [metadata_record(ctx, spec, "capability_matrix", payload)]
    for generator, info in payload.get("generators", {}).items():
        row = metadata_record(ctx, spec, f"capability_{generator}", info)
        row["generator"] = generator
        rows.append(row)
    return rows


def gate_record(ctx: Any, spec: Any, backend: str, generator: str, distribution: str, n: int, validation: dict[str, Any]) -> dict[str, Any]:
    record = base_record(ctx, spec, backend, "gate", generator, distribution, n, validation=validation)
    record["formal_result"] = validation.get("status") == "pass"
    return record


def unsupported_record(
    ctx: Any,
    spec: Any,
    backend: str,
    reason: str,
    *,
    generator: str = "not_applicable",
    distribution: str = "not_applicable",
    n: int = 0,
    parameters: dict[str, Any] | None = None,
    comparison_key: str | None = None,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    record = base_record(ctx, spec, backend, "unsupported", generator, distribution, n, validation=unsupported(reason), parameters=parameters)
    record.update(
        {
            "comparison_key": comparison_key,
            "is_baseline": False,
            "baseline_id": baseline_id,
            "formal_result": False,
            "audit_flags": ["unsupported_backend"],
        }
    )
    return record


def error_record(
    ctx: Any,
    spec: Any,
    backend: str,
    exc: BaseException,
    *,
    generator: str = "not_applicable",
    distribution: str = "not_applicable",
    n: int = 0,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = base_record(ctx, spec, backend, "error", generator, distribution, n, validation=validation_error(exc), parameters=parameters)
    record.update({"formal_result": False, "audit_flags": ["runtime_error"]})
    return record


def finalize_records(records: list[dict[str, Any]]) -> None:
    apply_cross_record_gates(records)
    annotate_poisson_semantics(records)
    compute_speedups(records)


def apply_cross_record_gates(records: list[dict[str, Any]]) -> None:
    failed_sequence: set[tuple[str, str]] = set()
    failed_distribution: set[tuple[str, str, str]] = set()
    failed_basic: set[tuple[str, str, str]] = set()
    for record in records:
        if record.get("validation", {}).get("status") != "fail":
            continue
        key2 = _gate_key2(record)
        key3 = _gate_key3(record)
        if record.get("task_id") == "G2_REPRODUCIBILITY":
            failed_sequence.add(key2)
        elif record.get("task_id") == "G1_DISTRIBUTION_ROUGH_CHECK":
            failed_distribution.add(key3)
        elif record.get("task_id") == "G0_BASIC_CONTRACT":
            failed_basic.add(key3)

    gate_task_ids = {"G0_BASIC_CONTRACT", "G1_DISTRIBUTION_ROUGH_CHECK", "G2_REPRODUCIBILITY"}
    for record in records:
        if record.get("task_id") in gate_task_ids or not record.get("comparison_key"):
            continue
        record["gate_backend"] = canonical_gate_backend(str(record.get("backend")))
        key2 = _gate_key2(record)
        key3 = _gate_key3(record)
        failures: list[str] = []
        if key3 in failed_basic:
            failures.append("G0_BASIC_CONTRACT")
            add_audit_flag(record, "basic_contract_gate_failed")
        if key3 in failed_distribution:
            failures.append("G1_DISTRIBUTION_ROUGH_CHECK")
            add_audit_flag(record, "distribution_gate_failed")
        if key2 in failed_sequence:
            failures.append("G2_REPRODUCIBILITY")
            add_audit_flag(record, "sequence_semantics_gate_failed")
        if failures:
            record["formal_result"] = False
            record["cross_record_gate_failures"] = sorted(set(failures))
    annotate_diagnostic_source_gates(records, failed_basic, failed_distribution, failed_sequence)


def annotate_diagnostic_source_gates(
    records: list[dict[str, Any]],
    failed_basic: set[tuple[str, str, str]],
    failed_distribution: set[tuple[str, str, str]],
    failed_sequence: set[tuple[str, str]],
) -> None:
    for record in records:
        if record.get("result_role") != "diagnostic":
            continue
        if record.get("backend") not in DIAGNOSTIC_FLAGRAND_BACKENDS:
            continue
        source_backend = "flagrand_public"
        generator = str(record.get("generator"))
        distribution = str(record.get("distribution"))
        component = str(record.get("diagnostic_component") or "")

        failures: list[str] = []
        if (source_backend, generator) in failed_sequence:
            failures.append("G2_REPRODUCIBILITY")
            add_audit_flag(record, "source_sequence_semantics_gate_failed")
        if (source_backend, generator, "raw32") in failed_basic:
            failures.append("G0_BASIC_CONTRACT")
            add_audit_flag(record, "source_basic_contract_gate_failed")
        if component in DIAGNOSTIC_DISTRIBUTION_COMPONENTS and (source_backend, generator, distribution) in failed_distribution:
            failures.append("G1_DISTRIBUTION_ROUGH_CHECK")
            add_audit_flag(record, "source_distribution_gate_failed")

        source_failures = sorted(set(failures))
        record["source_gate_backend"] = source_backend
        record["source_gate_failures"] = source_failures
        record["source_semantic_status"] = "failed" if source_failures else "passed_known_gates"


def annotate_poisson_semantics(records: list[dict[str, Any]]) -> None:
    for record in records:
        if record.get("distribution") != "poisson_u32":
            continue
        lambda_val = _lambda_parameter(record)
        if lambda_val is None:
            continue
        if str(record.get("backend", "")).startswith("curand"):
            record["semantic_model"] = "strict_poisson"
            record["semantic_equivalence"] = "reference"
        elif _is_flagrand_record(record) and lambda_val >= 30.0:
            record["semantic_model"] = "poisson_normal_approximation"
            record["semantic_equivalence"] = "accepted_approximation"
            record["accepted_approximation"] = True
            add_audit_flag(record, "poisson_normal_approximation")
            limitations = record.setdefault("known_limitations", [])
            if POISSON_NORMAL_APPROXIMATION_NOTE not in limitations:
                limitations.append(POISSON_NORMAL_APPROXIMATION_NOTE)
        elif _is_flagrand_record(record):
            record["semantic_model"] = "poisson_inverse_cdf"
            record["semantic_equivalence"] = "intended_strict_poisson"


def _lambda_parameter(record: dict[str, Any]) -> float | None:
    params = record.get("parameters") or {}
    if "lambda" not in params:
        return None
    try:
        return float(params["lambda"])
    except (TypeError, ValueError):
        return None


def _is_flagrand_record(record: dict[str, Any]) -> bool:
    return str(record.get("backend", "")).startswith("flagrand")


def canonical_gate_backend(backend: str) -> str:
    if backend in PUBLIC_OUTPUT_BACKEND_GATE_ALIASES:
        return PUBLIC_OUTPUT_BACKEND_GATE_ALIASES[backend]
    if backend in CURAND_HOST_OUTPUT_BACKEND_GATE_ALIASES:
        return CURAND_HOST_OUTPUT_BACKEND_GATE_ALIASES[backend]
    return backend


def _gate_key2(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("gate_backend") or canonical_gate_backend(str(record.get("backend")))), str(record.get("generator")))


def _gate_key3(record: dict[str, Any]) -> tuple[str, str, str]:
    gate_backend, generator = _gate_key2(record)
    return (gate_backend, generator, str(record.get("distribution")))


def add_audit_flag(record: dict[str, Any], flag: str) -> None:
    flags = record.get("audit_flags")
    if not isinstance(flags, list):
        flags = []
        record["audit_flags"] = flags
    if flag not in flags:
        flags.append(flag)


def compute_speedups(records: list[dict[str, Any]]) -> None:
    baselines: dict[str, dict[str, float | None]] = {}
    seen_baseline_keys: set[str] = set()
    for record in records:
        record["speedup_gpu_vs_baseline"] = None
        record["speedup_wall_vs_baseline"] = None
        record["speedup_baseline_formal"] = False
    for record in records:
        key = record.get("comparison_key")
        if not key or not record.get("is_baseline"):
            continue
        seen_baseline_keys.add(key)
        if not record.get("formal_result"):
            continue
        gpu = record.get("median_gpu_us")
        wall = record.get("median_wall_sync_us")
        if gpu is not None or wall is not None:
            baselines[key] = {"gpu": gpu, "wall": wall}
    for record in records:
        key = record.get("comparison_key")
        base = baselines.get(key)
        record["speedup_baseline_formal"] = bool(base)
        if key and base and not record.get("formal_result"):
            if not record.get("is_baseline"):
                add_audit_flag(record, "record_not_formal")
            continue
        if not key or not base:
            if key in seen_baseline_keys and not record.get("is_baseline"):
                add_audit_flag(record, "baseline_not_formal")
            continue
        gpu = record.get("median_gpu_us")
        wall = record.get("median_wall_sync_us")
        record["speedup_gpu_vs_baseline"] = _ratio(base.get("gpu"), gpu)
        record["speedup_wall_vs_baseline"] = _ratio(base.get("wall"), wall)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _estimate_output_bytes(distribution: str, n: int) -> int | None:
    if "raw64" in distribution:
        return n * 8
    if "dropout" in distribution:
        return n * 5
    if "threshold" in distribution:
        return n
    if distribution == "metadata":
        return None
    return n * 4


def _dtype_name(distribution: str) -> str:
    if "raw64" in distribution:
        return "uint64/int64_storage"
    if "raw32" in distribution or "poisson" in distribution:
        return "uint32/int32_storage"
    if "threshold" in distribution or "mask" in distribution:
        return "uint8"
    if distribution in {"metadata", "lifecycle", "generate_seeds", "error_case"}:
        return "not_applicable"
    return "float32"
