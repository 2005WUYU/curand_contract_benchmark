from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_summary(
    path: Path,
    *,
    records: list[dict[str, Any]],
    capability_matrix: dict[str, Any],
    environment: dict[str, Any],
    shard_process_failures: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            summarize_records(
                records,
                capability_matrix=capability_matrix,
                environment=environment,
                shard_process_failures=shard_process_failures,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def summarize_records(
    records: list[dict[str, Any]],
    *,
    capability_matrix: dict[str, Any],
    environment: dict[str, Any],
    shard_process_failures: list[str] | None = None,
) -> dict[str, Any]:
    claim_records = [record for record in records if not _is_diagnostic(record)]
    diagnostic_records = [record for record in records if _is_diagnostic(record)]
    status_counts = Counter(str(record.get("validation", {}).get("status")) for record in records)
    formal_counts = Counter(str(record.get("formal_result")) for record in records)
    audit_flags = Counter(flag for record in records for flag in (record.get("audit_flags") or []))
    failures = [record for record in records if record.get("validation", {}).get("status") == "fail"]
    unsupported_rows = [record for record in records if record.get("validation", {}).get("status") == "unsupported"]
    formal_speedups = [
        record
        for record in claim_records
        if not record.get("is_baseline")
        and record.get("formal_result")
        and record.get("speedup_baseline_formal")
        and record.get("speedup_gpu_vs_baseline") is not None
    ]
    return {
        "schema_version": 3,
        "record_count": len(records),
        "claim_record_count": len(claim_records),
        "diagnostic_record_count": len(diagnostic_records),
        "commit": _commit_from_environment(environment),
        "validation_scope": "rough_gate",
        "status_counts": dict(status_counts),
        "formal_counts": dict(formal_counts),
        "audit_flag_counts": dict(audit_flags.most_common()),
        "run_health": _run_health(records, shard_process_failures=shard_process_failures),
        "task_counts": dict(Counter(record.get("task_id") for record in records)),
        "diagnostic_counts_by_component": dict(Counter(record.get("diagnostic_component") for record in diagnostic_records)),
        "diagnostic_counts_by_distribution": dict(Counter(record.get("distribution") for record in diagnostic_records)),
        "diagnostic_validation_counts": dict(Counter(str(record.get("validation", {}).get("status")) for record in diagnostic_records)),
        "diagnostic_source_status_counts": dict(Counter(str(record.get("source_semantic_status")) for record in diagnostic_records if record.get("source_semantic_status"))),
        "semantic_model_counts": dict(Counter(str(record.get("semantic_model")) for record in records if record.get("semantic_model"))),
        "semantic_equivalence_counts": dict(Counter(str(record.get("semantic_equivalence")) for record in records if record.get("semantic_equivalence"))),
        "failure_counts_by_task": dict(Counter(record.get("task_id") for record in failures)),
        "unsupported_counts_by_task": dict(Counter(record.get("task_id") for record in unsupported_rows)),
        "unsupported_counts_by_backend": dict(Counter(record.get("backend") for record in unsupported_rows)),
        "cross_record_gate_failures": _cross_record_gate_failure_counts(records),
        "gate_backend_alias_counts": _gate_backend_alias_counts(records),
        "formal_speedups": _speedup_summary(formal_speedups),
        "formal_speedups_by_task": _speedups_by_task(formal_speedups),
        "formal_speedups_by_semantic_equivalence": _speedups_by_semantic_equivalence(formal_speedups),
        "failures": [_failure_summary(record) for record in failures],
        "capabilities": {
            "device_api_extension": _capability_support(capability_matrix.get("device_api_extension", {})),
            "curanddx": _capability_support(capability_matrix.get("curanddx", {})),
        },
        "shard_process_failures": shard_process_failures or [],
    }


def _run_health(records: list[dict[str, Any]], *, shard_process_failures: list[str] | None = None) -> dict[str, Any]:
    gate_task_ids = {"G0_BASIC_CONTRACT", "G1_DISTRIBUTION_ROUGH_CHECK", "G2_REPRODUCIBILITY"}
    validation_fail_count = sum(1 for record in records if record.get("validation", {}).get("status") == "fail")
    runtime_error_count = sum(1 for record in records if "runtime_error" in (record.get("audit_flags") or []))
    formal_gate_leak_count = sum(
        1
        for record in records
        if record.get("formal_result") and _has_gate_failure(record)
    )
    required_gate_failures = [
        record
        for record in records
        if record.get("task_id") in gate_task_ids and record.get("validation", {}).get("status") == "fail"
    ]
    cross_gate_failure_count = sum(len(record.get("cross_record_gate_failures") or []) for record in records)
    diagnostic_source_failure_count = sum(1 for record in records if record.get("source_semantic_status") == "failed")
    shard_process_failure_count = len(shard_process_failures or [])
    status = "ok"
    if runtime_error_count or formal_gate_leak_count or validation_fail_count or shard_process_failure_count:
        status = "needs_attention"
    return {
        "status": status,
        "status_reason": _run_health_reason(status),
        "validation_fail_count": validation_fail_count,
        "runtime_error_count": runtime_error_count,
        "formal_gate_leak_count": formal_gate_leak_count,
        "required_gate_failure_count": len(required_gate_failures),
        "required_gate_failures_by_task": dict(Counter(record.get("task_id") for record in required_gate_failures)),
        "cross_record_gate_failure_count": cross_gate_failure_count,
        "diagnostic_source_failure_count": diagnostic_source_failure_count,
        "shard_process_failure_count": shard_process_failure_count,
        "shard_process_failures": shard_process_failures or [],
    }


def _speedup_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(record["speedup_gpu_vs_baseline"]) for record in records)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "max": values[-1],
    }


def _speedups_by_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id"))].append(record)
    return {task_id: _speedup_summary(rows) for task_id, rows in sorted(grouped.items())}


def _speedups_by_semantic_equivalence(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("semantic_equivalence") or "unspecified")
        grouped[key].append(record)
    return {key: _speedup_summary(rows) for key, rows in sorted(grouped.items())}


def _has_gate_failure(record: dict[str, Any]) -> bool:
    flags = record.get("audit_flags") or []
    if any("gate_failed" in str(flag) for flag in flags):
        return True
    return bool(record.get("cross_record_gate_failures"))


def _run_health_reason(status: str) -> str:
    if status == "needs_attention":
        return "Validation failures, runtime errors, shard failures, or formal records with gate failures are present."
    return "No validation failures, runtime errors, shard failures, or required gate failures were found."


def _cross_record_gate_failure_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        for failure in record.get("cross_record_gate_failures") or []:
            counts[str(failure)] += 1
    return dict(counts)


def _gate_backend_alias_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        backend = record.get("backend")
        gate_backend = record.get("gate_backend")
        if backend and gate_backend and backend != gate_backend:
            counts[f"{backend}->{gate_backend}"] += 1
    return dict(counts)


def _failure_summary(record: dict[str, Any]) -> dict[str, Any]:
    checks = record.get("validation", {}).get("checks") or {}
    failed_checks = sorted(key for key, value in checks.items() if isinstance(value, bool) and not value)
    return {
        "task_id": record.get("task_id"),
        "backend": record.get("backend"),
        "generator": record.get("generator"),
        "distribution": record.get("distribution"),
        "N": record.get("N"),
        "parameters": record.get("parameters") or {},
        "failed_checks": failed_checks,
        "error_type": record.get("validation", {}).get("error_type"),
        "error": record.get("validation", {}).get("error"),
    }


def _capability_support(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(row.get("available")),
        "available_on_any_shard": row.get("available_on_any_shard"),
        "available_on_all_shards": row.get("available_on_all_shards"),
        "headers_available": row.get("headers_available"),
        "header_paths": row.get("header_paths"),
        "extension_available": row.get("extension_available"),
        "extension_build_dir": row.get("extension_build_dir"),
        "extension_symbols": row.get("extension_symbols"),
        "unsupported_reason": row.get("unsupported_reason"),
    }


def _commit_from_environment(environment: dict[str, Any]) -> str | None:
    git = environment.get("git") or {}
    if isinstance(git, dict) and git.get("commit"):
        return str(git.get("commit"))
    if environment.get("launcher_git_commit"):
        return str(environment.get("launcher_git_commit"))
    return None


def _is_diagnostic(record: dict[str, Any]) -> bool:
    return record.get("result_role") == "diagnostic" or record.get("family") == "diagnostic"
