from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from contract_benchmark.reporting import write_csv, write_json, write_jsonl, write_report, write_task_registry  # noqa: E402
from contract_benchmark.spec import specs_for_groups  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contract benchmark shards across multiple visible GPUs.")
    parser.add_argument("--profile", default="h20")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return parser.parse_args()


def main() -> int:
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
        "shards": [],
    }

    processes: list[tuple[int, Path, subprocess.Popen[bytes], Any]] = []
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
        manifest["shards"].append(
            {
                "shard": shard_index,
                "gpu": gpu_id,
                "tasks": task_ids,
                "run_dir": str(run_dir),
                "log": str(log_path),
            }
        )
        log_file = log_path.open("wb")
        print(f"[h20-parallel] launch shard={shard_index} gpu={gpu_id} tasks={len(task_ids)} log={log_path}", flush=True)
        processes.append((shard_index, run_dir, subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT), log_file))

    write_json(root_dir / "parallel_manifest.json", manifest)

    failures: list[str] = []
    for shard_index, run_dir, process, log_file in processes:
        rc = process.wait()
        log_file.close()
        if rc != 0:
            failures.append(f"shard {shard_index} failed with exit code {rc}; see {root_dir / f'shard_{shard_index:02d}.log'}")
        else:
            print(f"[h20-parallel] shard={shard_index} done run_dir={run_dir}", flush=True)

    if failures:
        raise SystemExit("\n".join(failures))

    records: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None
    shard_environments: list[dict[str, Any]] = []
    shard_capability_matrices: list[dict[str, Any]] = []
    for _, run_dir, _, _ in processes:
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
        "shard_environments": shard_environments,
    }
    capability_matrix = _merge_capability_matrices(shard_capability_matrices)

    write_json(root_dir / "environment.json", environment)
    write_json(root_dir / "capability_matrix.json", capability_matrix)
    write_task_registry(root_dir / "task_registry.json", selected_specs)
    write_jsonl(root_dir / "results.jsonl", records)
    write_csv(root_dir / "results.csv", records)
    write_report(
        root_dir / "REPORT.md",
        records=records,
        specs=selected_specs,
        environment=environment,
        capability_matrix=capability_matrix,
        h20_reference=None,
    )

    pass_count = sum(1 for r in records if r.get("validation", {}).get("status") == "pass")
    fail_count = sum(1 for r in records if r.get("validation", {}).get("status") == "fail")
    unsupported_count = sum(1 for r in records if r.get("validation", {}).get("status") == "unsupported")
    print(f"[h20-parallel] results: {root_dir}", flush=True)
    print(
        f"[h20-parallel] tasks={len(selected_specs)} records={len(records)} "
        f"pass={pass_count} fail={fail_count} unsupported={unsupported_count}",
        flush=True,
    )
    print(f"[h20-parallel] report: {root_dir / 'REPORT.md'}", flush=True)
    return 0


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
    reasons = [
        str(row.get("unsupported_reason"))
        for row in rows
        if row.get("unsupported_reason")
    ]
    merged = copy.deepcopy(rows[0]) if rows else {}
    merged["available"] = bool(available_values) and all(available_values)
    merged["available_on_any_shard"] = any(available_values)
    merged["available_on_all_shards"] = bool(available_values) and all(available_values)
    merged["per_shard"] = rows
    if not merged["available_on_all_shards"]:
        merged["unsupported_reason"] = "; ".join(sorted(set(reasons))) or "not available on every shard"
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
