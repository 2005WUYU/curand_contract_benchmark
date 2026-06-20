from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_summary(path: Path, *, records: list[dict[str, Any]], capability_matrix: dict[str, Any], environment: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_records(records, capability_matrix=capability_matrix, environment=environment), indent=2, sort_keys=True), encoding="utf-8")


def summarize_records(records: list[dict[str, Any]], *, capability_matrix: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("validation", {}).get("status")) for record in records)
    formal_counts = Counter(str(record.get("formal_result")) for record in records)
    audit_flags = Counter(flag for record in records for flag in (record.get("audit_flags") or []))
    failures = [record for record in records if record.get("validation", {}).get("status") == "fail"]
    unsupported_rows = [record for record in records if record.get("validation", {}).get("status") == "unsupported"]
    formal_speedups = [
        record
        for record in records
        if not record.get("is_baseline")
        and record.get("formal_result")
        and record.get("speedup_baseline_formal")
        and record.get("speedup_gpu_vs_baseline") is not None
    ]
    return {
        "schema_version": 1,
        "record_count": len(records),
        "commit": _commit_from_environment(environment),
        "validation_scope": "rough_gate",
        "status_counts": dict(status_counts),
        "formal_counts": dict(formal_counts),
        "audit_flag_counts": dict(audit_flags.most_common()),
        "task_counts": dict(Counter(record.get("task_id") for record in records)),
        "failure_counts_by_task": dict(Counter(record.get("task_id") for record in failures)),
        "unsupported_counts_by_task": dict(Counter(record.get("task_id") for record in unsupported_rows)),
        "unsupported_counts_by_backend": dict(Counter(record.get("backend") for record in unsupported_rows)),
        "cross_record_gate_failures": _cross_record_gate_failure_counts(records),
        "formal_speedups": _speedup_summary(formal_speedups),
        "formal_speedups_by_task": _speedups_by_task(formal_speedups),
        "failures": [_failure_summary(record) for record in failures],
        "capabilities": {
            "device_api_extension": _capability_support(capability_matrix.get("device_api_extension", {})),
            "curanddx": _capability_support(capability_matrix.get("curanddx", {})),
        },
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


def _cross_record_gate_failure_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        for failure in record.get("cross_record_gate_failures") or []:
            counts[str(failure)] += 1
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
        "unsupported_reason": row.get("unsupported_reason"),
    }


def _commit_from_environment(environment: dict[str, Any]) -> str | None:
    git = environment.get("git") or {}
    if isinstance(git, dict) and git.get("commit"):
        return str(git.get("commit"))
    if environment.get("launcher_git_commit"):
        return str(environment.get("launcher_git_commit"))
    return None
