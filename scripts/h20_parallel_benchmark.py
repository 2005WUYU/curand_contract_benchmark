from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from contract_benchmark.runtime_env import configure_writable_cache  # noqa: E402


configure_writable_cache(REPO_ROOT)

from contract_benchmark.reporting import write_csv, write_json, write_jsonl, write_report, write_task_registry  # noqa: E402
from contract_benchmark.runner import finalize_records  # noqa: E402
from contract_benchmark.spec import specs_for_groups  # noqa: E402
from contract_benchmark.summary import summarize_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contract benchmark shards across multiple visible GPUs.")
    parser.add_argument("--profile", default="h20")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return parser.parse_args()


def main() -> int:
    launcher_started_unix = time.time()
    args = parse_args()
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    selected_specs = specs_for_groups(groups or {"all"})
    if not selected_specs:
        raise SystemExit(f"No benchmark tasks matched groups={args.groups!r}")

    gpu_ids = _visible_gpu_ids(args.num_gpus)
    shard_count = min(max(1, args.num_gpus), len(gpu_ids), len(selected_specs))
    gpu_ids = gpu_ids[:shard_count]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = args.results_dir / f"{stamp}_{args.profile}_parallel_{shard_count}gpu"
    root_dir.mkdir(parents=True, exist_ok=False)

    shards: list[list[Any]] = [[] for _ in range(shard_count)]
    for index, spec in enumerate(selected_specs):
        shards[index % shard_count].append(spec)

    manifest = {
        "profile": args.profile,
        "groups": args.groups,
        "requested_num_gpus": args.num_gpus,
        "shard_count": shard_count,
        "gpu_ids": gpu_ids,
        "root_dir": str(root_dir),
        "launcher_started_unix": launcher_started_unix,
        "shards": [],
    }

    processes: list[tuple[int, Path, subprocess.Popen[bytes], Any, dict[str, Any]]] = []
    for shard_index, specs in enumerate(shards):
        if not specs:
            continue
        gpu_id = gpu_ids[shard_index]
        task_ids = [spec.task_id for spec in specs]
        run_dir = root_dir / f"shard_{shard_index:02d}_gpu_{_safe_name(gpu_id)}"
        log_path = root_dir / f"shard_{shard_index:02d}.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "run_benchmark.py"),
            "--profile",
            args.profile,
            "--groups",
            ",".join(task_ids),
            "--run-dir",
            str(run_dir),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["CURAND_CONTRACT_SHARD"] = str(shard_index)
        env["CURAND_CONTRACT_SHARD_COUNT"] = str(shard_count)
        env["HOME"] = str(REPO_ROOT)
        env["XDG_CACHE_HOME"] = str(root_dir / ".cache" / f"shard_{shard_index:02d}")
        env["TRITON_CACHE_DIR"] = str(root_dir / ".cache" / "triton" / f"shard_{shard_index:02d}")
        env["TORCH_EXTENSIONS_DIR"] = str(root_dir / ".cache" / "torch_extensions")
        Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
        shard_manifest = {
            "shard": shard_index,
            "gpu": gpu_id,
            "tasks": task_ids,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "triton_cache_dir": env["TRITON_CACHE_DIR"],
            "launch_started_unix": time.time(),
        }
        manifest["shards"].append(shard_manifest)
        log_file = log_path.open("wb")
        print(f"[h20-parallel] launch shard={shard_index} gpu={gpu_id} tasks={len(task_ids)} log={log_path}", flush=True)
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((shard_index, run_dir, process, log_file, shard_manifest))

    write_json(root_dir / "parallel_manifest.json", manifest)

    failures: list[str] = []
    for shard_index, run_dir, process, log_file, shard_manifest in processes:
        rc = process.wait()
        shard_ended_unix = time.time()
        log_file.close()
        shard_manifest["exit_code"] = rc
        shard_manifest["ended_unix"] = shard_ended_unix
        shard_manifest["elapsed_seconds"] = shard_ended_unix - float(shard_manifest["launch_started_unix"])
        if rc != 0:
            failures.append(f"shard {shard_index} failed with exit code {rc}; see {root_dir / f'shard_{shard_index:02d}.log'}")
        else:
            print(f"[h20-parallel] shard={shard_index} done elapsed_seconds={shard_manifest['elapsed_seconds']:.3f} run_dir={run_dir}", flush=True)

    shards_ended_unix = time.time()
    manifest["shards_ended_unix"] = shards_ended_unix
    manifest["shard_phase_elapsed_seconds"] = shards_ended_unix - launcher_started_unix
    write_json(root_dir / "parallel_manifest.json", manifest)

    records: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None
    shard_environments: list[dict[str, Any]] = []
    shard_capability_matrices: list[dict[str, Any]] = []
    for _, run_dir, _, _, _ in processes:
        if environment is None and (run_dir / "environment.json").exists():
            environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        if (run_dir / "environment.json").exists():
            shard_environments.append(json.loads((run_dir / "environment.json").read_text(encoding="utf-8")))
        if (run_dir / "capability_matrix.json").exists():
            shard_capability_matrices.append(json.loads((run_dir / "capability_matrix.json").read_text(encoding="utf-8")))
        jsonl = run_dir / "results.jsonl"
        if jsonl.exists():
            with jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

    if environment is None:
        environment = {"profile": args.profile}
    environment["parallel_launcher"] = {
        "shard_count": shard_count,
        "gpu_ids": gpu_ids,
        "groups": args.groups,
        "launcher_started_unix": launcher_started_unix,
        "shards_ended_unix": shards_ended_unix,
        "shard_phase_elapsed_seconds": shards_ended_unix - launcher_started_unix,
        "shards": manifest["shards"],
        "shard_environments": shard_environments,
    }
    capability_matrix = _merge_capability_matrices(shard_capability_matrices)
    finalize_records(records)

    aggregation_ended_unix = time.time()
    manifest["aggregation_ended_unix"] = aggregation_ended_unix
    manifest["elapsed_seconds_before_output_write"] = aggregation_ended_unix - launcher_started_unix
    environment["parallel_launcher"]["aggregation_ended_unix"] = aggregation_ended_unix
    environment["parallel_launcher"]["elapsed_seconds_before_output_write"] = aggregation_ended_unix - launcher_started_unix
    summary = summarize_records(
        records,
        capability_matrix=capability_matrix,
        environment=environment,
        shard_process_failures=failures,
    )

    write_json(root_dir / "parallel_manifest.json", manifest)
    write_json(root_dir / "environment.json", environment)
    write_json(root_dir / "capability_matrix.json", capability_matrix)
    write_task_registry(root_dir / "task_registry.json", selected_specs)
    write_jsonl(root_dir / "results.jsonl", records)
    write_csv(root_dir / "results.csv", records)
    write_json(root_dir / "summary.json", summary)
    write_report(
        root_dir / "REPORT.md",
        records=records,
        specs=selected_specs,
        environment=environment,
        capability_matrix=capability_matrix,
        h20_reference=None,
        run_summary=summary,
    )

    pass_count = sum(1 for r in records if r.get("validation", {}).get("status") == "pass")
    fail_count = sum(1 for r in records if r.get("validation", {}).get("status") == "fail")
    unsupported_count = sum(1 for r in records if r.get("validation", {}).get("status") == "unsupported")
    print(f"[h20-parallel] results: {root_dir}", flush=True)
    print(
        f"[h20-parallel] tasks={len(selected_specs)} records={len(records)} "
        f"pass={pass_count} fail={fail_count} unsupported={unsupported_count} "
        f"health={summary.get('run_health', {}).get('status')}",
        flush=True,
    )
    for failure in failures:
        print(f"[h20-parallel] issue: {failure}", file=sys.stderr, flush=True)
    print(f"[h20-parallel] elapsed_seconds_before_output_write={aggregation_ended_unix - launcher_started_unix:.3f}", flush=True)
    print(f"[h20-parallel] report: {root_dir / 'REPORT.md'}", flush=True)
    return 0 if _summary_exit_ok(summary) else 1


def _summary_exit_ok(summary: dict[str, object]) -> bool:
    status_counts = summary.get("status_counts")
    run_health = summary.get("run_health")
    fail_count = int(status_counts.get("fail", 0)) if isinstance(status_counts, dict) else 0
    health_status = run_health.get("status") if isinstance(run_health, dict) else None
    return fail_count == 0 and health_status == "ok"


def _visible_gpu_ids(requested: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        ids: list[str] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part and all(piece.isdigit() for piece in part.split("-", 1)):
                start_text, end_text = part.split("-", 1)
                ids.extend(str(value) for value in range(int(start_text), int(end_text) + 1))
            else:
                ids.append(part)
        if ids:
            return ids
    return [str(index) for index in range(max(1, requested))]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _merge_capability_matrices(matrices: list[dict[str, Any]]) -> dict[str, Any]:
    if not matrices:
        return {}
    merged = copy.deepcopy(matrices[0])
    merged["parallel_merged"] = True
    merged["shard_count"] = len(matrices)
    for key in ("device_api_extension", "curanddx"):
        rows = [matrix.get(key, {}) for matrix in matrices]
        if rows:
            merged[key] = _merge_support_rows(rows)
    merged["shards"] = [{"shard": index, "capability_matrix": matrix} for index, matrix in enumerate(matrices)]
    return merged


def _merge_support_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available_values = [bool(row.get("available")) for row in rows]
    header_values = [bool(row.get("headers_available")) for row in rows if row.get("headers_available") is not None]
    header_paths = sorted(
        {
            str(path)
            for row in rows
            for path in (row.get("header_paths") or [])
        }
    )
    reasons = [
        str(row.get("unsupported_reason"))
        for row in rows
        if row.get("unsupported_reason")
    ]
    merged = copy.deepcopy(rows[0]) if rows else {}
    merged["available"] = bool(available_values) and all(available_values)
    merged["available_on_any_shard"] = any(available_values)
    merged["available_on_all_shards"] = bool(available_values) and all(available_values)
    for list_key in (
        "build_dirs",
        "cuda_library_dirs",
        "cudart_candidates",
        "shared_objects",
        "missing_dependencies",
    ):
        values = sorted(
            {
                str(value)
                for row in rows
                for value in (row.get(list_key) or [])
            }
        )
        if values:
            merged[list_key] = values
    if header_values:
        merged["headers_available"] = any(header_values)
        merged["headers_available_on_all_shards"] = all(header_values)
    if header_paths:
        merged["header_paths"] = header_paths
    merged["per_shard"] = rows
    if not merged["available_on_all_shards"]:
        merged["unsupported_reason"] = "; ".join(sorted(set(reasons))) or "not available on every shard"
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
