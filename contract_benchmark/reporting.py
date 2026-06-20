from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from contract_benchmark.spec import TaskSpec


def make_run_dir(base_dir: Path, profile: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{stamp}_{profile}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "claim_id",
        "backend",
        "gate_backend",
        "api_surface",
        "generator",
        "distribution",
        "ordering",
        "N",
        "parameters",
        "validation_status",
        "validation_scope",
        "unsupported_reason",
        "median_gpu_us",
        "median_wall_sync_us",
        "median_cpu_enqueue_us",
        "items_per_second",
        "bytes_per_second",
        "speedup_gpu_vs_baseline",
        "speedup_wall_vs_baseline",
        "speedup_baseline_formal",
        "formal_result",
        "cross_record_gate_failures",
        "audit_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            validation = record.get("validation", {})
            row = {field: record.get(field) for field in fields}
            row["validation_status"] = validation.get("status")
            row["unsupported_reason"] = validation.get("unsupported_reason")
            row["parameters"] = json.dumps(record.get("parameters", {}), ensure_ascii=False, sort_keys=True)
            row["cross_record_gate_failures"] = json.dumps(record.get("cross_record_gate_failures", []), ensure_ascii=False)
            row["audit_flags"] = json.dumps(record.get("audit_flags", []), ensure_ascii=False)
            writer.writerow(row)


def write_task_registry(path: Path, specs: list[TaskSpec]) -> None:
    write_json(path, [spec.to_record() for spec in specs])


def write_report(
    path: Path,
    *,
    records: list[dict[str, Any]],
    specs: list[TaskSpec],
    environment: dict[str, Any],
    capability_matrix: dict[str, Any],
    h20_reference: dict[str, Any] | None,
) -> None:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)

    lines: list[str] = [
        "# Contract Benchmark Report",
        "",
        "This report follows the two planning documents as task contracts: claim first, task second, timing boundary explicit, validation before performance.",
        "",
        "## Environment",
        "",
        f"- profile: `{environment.get('profile')}`",
        f"- torch: `{environment.get('torch_version')}`",
        f"- cuda available: `{environment.get('cuda_available')}`",
        f"- gpu: `{environment.get('gpu_name')}`",
        f"- cuRAND version: `{environment.get('curand', {}).get('version')}`",
        "",
        "## Coverage",
        "",
    ]
    required = [spec for spec in specs if "must" in spec.required_by]
    executed_task_ids = set(by_task)
    for spec in required:
        status = "executed" if spec.task_id in executed_task_ids else "not_run"
        lines.append(f"- `{spec.task_id}`: {status} ({spec.required_by})")

    lines.extend(["", "## Capability Matrix", ""])
    device_ext = capability_matrix.get("device_api_extension", {})
    curanddx = capability_matrix.get("curanddx", {})
    lines.append(f"- legacy Device API extension: `{_support_state(device_ext)}`")
    if device_ext.get("unsupported_reason"):
        lines.append(f"  reason: {device_ext.get('unsupported_reason')}")
    lines.append(f"- cuRANDDx: `{_support_state(curanddx)}`")
    if curanddx.get("unsupported_reason"):
        lines.append(f"  reason: {curanddx.get('unsupported_reason')}")

    lines.extend(["", "## Task Results", ""])
    lines.append(
        "| task | backend | generator | distribution | N | gpu_us | wall_us | speedup_gpu | speedup_wall | formal | baseline_formal | validation | audit |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for task_id in sorted(by_task):
        for record in by_task[task_id]:
            validation = record.get("validation", {})
            lines.append(
                "| {task} | {backend} | {generator} | {distribution} | {n} | {gpu} | {wall} | {sg} | {sw} | {formal} | {baseline_formal} | {validation} | {audit} |".format(
                    task=task_id,
                    backend=record.get("backend"),
                    generator=record.get("generator"),
                    distribution=record.get("distribution"),
                    n=record.get("N"),
                    gpu=_fmt(record.get("median_gpu_us")),
                    wall=_fmt(record.get("median_wall_sync_us")),
                    sg=_fmt(record.get("speedup_gpu_vs_baseline")),
                    sw=_fmt(record.get("speedup_wall_vs_baseline")),
                    formal=record.get("formal_result"),
                    baseline_formal=record.get("speedup_baseline_formal"),
                    validation=validation.get("status"),
                    audit=",".join(record.get("audit_flags", [])),
                )
            )

    lines.extend(["", "## Operator Walkthrough: F1 Add Uniform", ""])
    lines.extend(_f1_walkthrough(records, h20_reference))

    lines.extend(["", "## Interpretation Guardrails", ""])
    guardrail_specs = {spec.task_id: spec for spec in specs}
    for task_id in sorted(by_task):
        spec = guardrail_specs.get(task_id)
        if spec is None:
            continue
        lines.append(f"- `{task_id}` can say: {spec.can_say}")
        lines.append(f"  cannot say: {spec.cannot_say}")

    lines.extend(["", "## Artifacts", ""])
    lines.extend(
        [
            "- `task_registry.json`: full claim/task/timing/validation contract.",
            "- `capability_matrix.json`: available and unsupported backends.",
            "- `results.jsonl`: full records with raw samples.",
            "- `results.csv`: compact table for spreadsheet use.",
            "- `environment.json`: versions and profile.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f1_walkthrough(records: list[dict[str, Any]], h20_reference: dict[str, Any] | None) -> list[str]:
    f1 = [r for r in records if r["task_id"] == "F1_ADD_UNIFORM"]
    if not f1:
        return ["F1 was not run in this profile."]
    baselines = [r for r in f1 if r.get("backend") == "curand_host_bulk_plus_consume" and r.get("formal_result")]
    candidates = [
        r
        for r in f1
        if r.get("backend") == "flagrand_fused_philox"
        and r.get("formal_result")
        and r.get("speedup_baseline_formal")
        and r.get("speedup_wall_vs_baseline") is not None
    ]
    if not baselines or not candidates:
        return ["F1 did not include both formal cuRAND Host bulk+consume and formal FlagRand fused rows with a formal baseline."]
    cand = min(candidates, key=lambda r: r.get("median_wall_sync_us") or float("inf"))
    base = next((r for r in baselines if r.get("comparison_key") == cand.get("comparison_key")), baselines[0])
    gpu_speed = cand.get("speedup_gpu_vs_baseline")
    wall_speed = cand.get("speedup_wall_vs_baseline")
    lines = [
        "Task contract: `y = x + alpha * (u - 0.5)` with random numbers consumed immediately.",
        f"- cuRAND path: Host API writes a temporary uniform buffer, then a consumer kernel reads it and writes `y`.",
        f"- FlagRand path: one fused Philox/Triton kernel generates `u` and writes `y` directly.",
        f"- local GPU-event speedup: `{_fmt(gpu_speed)}x`; local wall-sync speedup: `{_fmt(wall_speed)}x`.",
        "Allowed interpretation: this is an end-to-end solution result; any win includes avoiding a temporary random buffer and one consumer launch.",
        "Forbidden interpretation: this does not prove raw RNG generation is faster.",
    ]
    if h20_reference:
        lines.append("")
        lines.append("H20 old benchmark reference was detected and is included only as historical context, because it used a different task/timing contract.")
        for key, value in h20_reference.items():
            lines.append(f"- {key}: {value}")
    return lines


def _support_state(row: dict[str, Any]) -> str:
    if row.get("available_on_any_shard") and not row.get("available_on_all_shards"):
        return "partial"
    return "available" if row.get("available") else "unsupported"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
