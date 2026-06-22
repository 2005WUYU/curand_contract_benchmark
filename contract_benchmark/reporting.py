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
        "result_role",
        "backend",
        "gate_backend",
        "api_surface",
        "generator",
        "distribution",
        "diagnostic_component",
        "semantic_model",
        "semantic_equivalence",
        "accepted_approximation",
        "source_semantic_status",
        "source_gate_backend",
        "source_gate_failures",
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
            row["source_gate_failures"] = json.dumps(record.get("source_gate_failures", []), ensure_ascii=False)
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
    run_summary: dict[str, Any] | None = None,
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

    lines.extend(["", "## Run Health", ""])
    if run_summary is not None:
        lines.extend(_run_health_summary_lines(run_summary))
    else:
        lines.extend(_run_health_lines(records))

    semantic_lines = _semantic_notes(records)
    if semantic_lines:
        lines.extend(["", "## Semantic Notes", ""])
        lines.extend(semantic_lines)

    lines.extend(["", "## Capability Matrix", ""])
    device_ext = capability_matrix.get("device_api_extension", {})
    curanddx = capability_matrix.get("curanddx", {})
    lines.append(f"- legacy Device API extension: `{_support_state(device_ext)}`")
    if device_ext.get("module_file"):
        lines.append(f"  module_file: `{device_ext.get('module_file')}`")
    if device_ext.get("shared_objects"):
        lines.append(f"  shared_objects: `{device_ext.get('shared_objects')}`")
    if device_ext.get("missing_dependencies"):
        lines.append(f"  missing_dependencies: `{device_ext.get('missing_dependencies')}`")
    if device_ext.get("cudart_candidates"):
        lines.append(f"  cudart_candidates: `{device_ext.get('cudart_candidates')}`")
    if device_ext.get("extension_symbols"):
        lines.append(f"  extension_symbols: `{device_ext.get('extension_symbols')}`")
    if device_ext.get("unsupported_reason"):
        lines.append(f"  reason: {device_ext.get('unsupported_reason')}")
    lines.append(f"- cuRANDDx: `{_support_state(curanddx)}`")
    if curanddx.get("headers_available") is not None:
        lines.append(f"  headers_available: `{curanddx.get('headers_available')}`")
    if curanddx.get("extension_available") is not None:
        lines.append(f"  extension_available: `{curanddx.get('extension_available')}`")
    if curanddx.get("extension_build_dir"):
        lines.append(f"  extension_build_dir: `{curanddx.get('extension_build_dir')}`")
    if curanddx.get("extension_symbols"):
        lines.append(f"  extension_symbols: `{curanddx.get('extension_symbols')}`")
    if curanddx.get("header_paths"):
        lines.append(f"  headers: `{', '.join(str(path) for path in curanddx.get('header_paths', []))}`")
    if curanddx.get("unsupported_reason"):
        lines.append(f"  reason: {curanddx.get('unsupported_reason')}")

    lines.extend(["", "## Task Results", ""])
    lines.append(
        "| task | backend | generator | distribution | semantic | N | gpu_us | wall_us | speedup_gpu | speedup_wall | formal | baseline_formal | validation | audit |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for task_id in sorted(by_task):
        for record in by_task[task_id]:
            if _is_diagnostic(record):
                continue
            validation = record.get("validation", {})
            lines.append(
                "| {task} | {backend} | {generator} | {distribution} | {semantic} | {n} | {gpu} | {wall} | {sg} | {sw} | {formal} | {baseline_formal} | {validation} | {audit} |".format(
                    task=task_id,
                    backend=record.get("backend"),
                    generator=record.get("generator"),
                    distribution=record.get("distribution"),
                    semantic=_semantic_label(record),
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

    diagnostic_lines = _distribution_diagnostics(records)
    if diagnostic_lines:
        lines.extend(["", "## Distribution Diagnostics", ""])
        lines.extend(diagnostic_lines)

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
            "- `summary.json`: machine-readable run summary and diagnostic counts.",
            "- `results.jsonl`: full records with raw samples.",
            "- `results.csv`: compact table for spreadsheet use.",
            "- `environment.json`: versions and profile.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution_diagnostics(records: list[dict[str, Any]]) -> list[str]:
    diagnostic_records = [record for record in records if _is_diagnostic(record)]
    if not diagnostic_records:
        return []

    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in diagnostic_records:
        params = record.get("parameters") or {}
        lambda_value = params.get("lambda")
        lambda_label = "" if lambda_value is None else f"lambda={lambda_value}"
        key = (str(record.get("distribution")), int(record.get("N") or 0), lambda_label)
        component = str(record.get("diagnostic_component") or params.get("component") or record.get("backend"))
        grouped[key][component] = record

    lines = [
        "Diagnostic-only rows are excluded from formal speedup claims. They use selected sizes only, so the full record remains in `results.jsonl`/`results.csv` without expanding the main result table.",
        "",
        "| distribution | N | params | semantic | raw_only_gpu_us | transform_only_gpu_us | raw_plus_transform_gpu_us | public_api_gpu_us | validation | source_gates |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for (distribution, n, params), components in sorted(grouped.items()):
        lines.append(
            "| {distribution} | {n} | {params} | {semantic} | {raw} | {transform} | {combined} | {public} | {validation} | {source} |".format(
                distribution=distribution,
                n=n,
                params=params,
                semantic=_components_semantic_label(components),
                raw=_component_time(components.get("raw_only")),
                transform=_component_time(components.get("transform_only")),
                combined=_component_time(components.get("raw_plus_transform")),
                public=_component_time(components.get("public_api")),
                validation=_component_validation(components),
                source=_component_source_status(components),
            )
        )
    return lines


def _component_time(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return _fmt(record.get("median_gpu_us"))


def _component_validation(components: dict[str, dict[str, Any]]) -> str:
    failures = sorted(
        component
        for component, record in components.items()
        if record.get("validation", {}).get("status") != "pass"
    )
    if not failures:
        return "pass"
    return "fail:" + ",".join(failures)


def _semantic_notes(records: list[dict[str, Any]]) -> list[str]:
    approximate_poisson = [
        record
        for record in records
        if record.get("semantic_equivalence") == "accepted_approximation"
    ]
    if not approximate_poisson:
        return []
    counts = defaultdict(int)
    for record in approximate_poisson:
        counts[str(record.get("task_id"))] += 1
    count_text = ", ".join(f"`{task}`={count}" for task, count in sorted(counts.items()))
    return [
        "FlagRand Poisson rows with `lambda >= 30` are marked as `accepted_approximation:poisson_normal_approximation`.",
        "These rows are useful for approximate-Poisson engineering comparisons, but they should not be described as strict cuRAND-equivalent Poisson.",
        f"- approximate rows by task: {count_text}",
    ]


def _semantic_label(record: dict[str, Any]) -> str:
    semantic = record.get("semantic_equivalence")
    model = record.get("semantic_model")
    if semantic and model:
        return f"{semantic}:{model}"
    if semantic:
        return str(semantic)
    if model:
        return str(model)
    return ""


def _components_semantic_label(components: dict[str, dict[str, Any]]) -> str:
    labels = sorted({_semantic_label(record) for record in components.values() if _semantic_label(record)})
    return ",".join(labels)


def _component_source_status(components: dict[str, dict[str, Any]]) -> str:
    failures = []
    for component, record in sorted(components.items()):
        if record.get("source_semantic_status") != "failed":
            continue
        gates = ",".join(record.get("source_gate_failures") or [])
        failures.append(f"{component}:{gates}")
    if failures:
        return "failed:" + ";".join(failures)
    if any(record.get("source_semantic_status") for record in components.values()):
        return "passed_known_gates"
    return ""


def _run_health_lines(records: list[dict[str, Any]]) -> list[str]:
    gate_task_ids = {"G0_BASIC_CONTRACT", "G1_DISTRIBUTION_ROUGH_CHECK", "G2_REPRODUCIBILITY"}
    validation_fail_count = sum(1 for record in records if record.get("validation", {}).get("status") == "fail")
    runtime_error_count = sum(1 for record in records if "runtime_error" in (record.get("audit_flags") or []))
    formal_gate_leak_count = sum(1 for record in records if record.get("formal_result") and _has_gate_failure(record))
    required_gate_failures = [
        record
        for record in records
        if record.get("task_id") in gate_task_ids and record.get("validation", {}).get("status") == "fail"
    ]
    if validation_fail_count or runtime_error_count or formal_gate_leak_count:
        status = "needs_attention"
    else:
        status = "ok"
    lines = [
        f"- status: `{status}`",
        f"- validation failures: `{validation_fail_count}`",
        f"- runtime errors: `{runtime_error_count}`",
        f"- formal gate leaks: `{formal_gate_leak_count}`",
        f"- required gate failures: `{len(required_gate_failures)}`",
    ]
    if required_gate_failures:
        counts = defaultdict(int)
        for record in required_gate_failures:
            counts[str(record.get("task_id"))] += 1
        lines.append("- required gate failures by task: " + ", ".join(f"`{task}`={count}" for task, count in sorted(counts.items())))
    return lines


def _run_health_summary_lines(summary: dict[str, Any]) -> list[str]:
    run_health = summary.get("run_health") or {}
    lines = [
        f"- status: `{run_health.get('status')}`",
        f"- validation failures: `{run_health.get('validation_fail_count', 0)}`",
        f"- runtime errors: `{run_health.get('runtime_error_count', 0)}`",
        f"- formal gate leaks: `{run_health.get('formal_gate_leak_count', 0)}`",
        f"- required gate failures: `{run_health.get('required_gate_failure_count', 0)}`",
        f"- shard process failures: `{run_health.get('shard_process_failure_count', 0)}`",
    ]
    failures_by_task = run_health.get("required_gate_failures_by_task") or {}
    if failures_by_task:
        lines.append("- required gate failures by task: " + ", ".join(f"`{task}`={count}" for task, count in sorted(failures_by_task.items())))
    shard_failures = run_health.get("shard_process_failures") or []
    for failure in shard_failures:
        lines.append(f"- shard failure: {failure}")
    return lines


def _has_gate_failure(record: dict[str, Any]) -> bool:
    flags = record.get("audit_flags") or []
    if any("gate_failed" in str(flag) for flag in flags):
        return True
    return bool(record.get("cross_record_gate_failures"))


def _is_diagnostic(record: dict[str, Any]) -> bool:
    return record.get("result_role") == "diagnostic" or record.get("family") == "diagnostic"


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
