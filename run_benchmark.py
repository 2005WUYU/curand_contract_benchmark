from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
LOCAL_SRC_ROOT = BENCHMARK_ROOT / "src"
PARENT_SRC_ROOT = REPO_ROOT / "src"
for path in (BENCHMARK_ROOT, LOCAL_SRC_ROOT, PARENT_SRC_ROOT):
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)

from contract_benchmark.reporting import (  # noqa: E402
    make_run_dir,
    write_csv,
    write_json,
    write_jsonl,
    write_report,
    write_task_registry,
)
from contract_benchmark.runner import (  # noqa: E402
    BenchmarkContext,
    PROFILES,
    collect_environment,
    run_specs,
)
from contract_benchmark.spec import build_task_specs, specs_for_groups  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contract-style FlagRand vs cuRAND benchmark")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="local_smoke")
    parser.add_argument(
        "--groups",
        default="stage0,stage1,stage2,stage3,stage4",
        help="Comma-separated task ids, families, stage0..stage4, or all.",
    )
    parser.add_argument("--results-dir", type=Path, default=BENCHMARK_ROOT / "results")
    parser.add_argument("--run-dir", type=Path, default=None, help="Exact output directory. Intended for sharded/parallel launchers.")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--h20-reference", type=Path, default=REPO_ROOT / "20260615-185511-h20-full-fast")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_specs = build_task_specs()
    if args.list_tasks:
        for spec in all_specs:
            print(f"{spec.task_id}\tstage={spec.stage}\tfamily={spec.family}\trequired={spec.required_by}")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    profile = PROFILES[args.profile]
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    selected_specs = specs_for_groups(groups or {"all"})
    selected_map = {spec.task_id: spec for spec in selected_specs}

    if args.run_dir is not None:
        results_dir = args.run_dir
        results_dir.mkdir(parents=True, exist_ok=False)
    else:
        results_dir = make_run_dir(args.results_dir, args.profile)
    print(f"[contract-benchmark] results: {results_dir}")
    print(f"[contract-benchmark] profile={args.profile} tasks={len(selected_specs)}")

    ctx = BenchmarkContext(
        repo_root=REPO_ROOT,
        benchmark_root=BENCHMARK_ROOT,
        profile=profile,
        specs=selected_map,
        device=torch.device("cuda"),
    )

    environment = collect_environment(args.profile)
    records, cap_matrix = run_specs(ctx, selected_specs)
    h20_reference = _load_h20_reference(args.h20_reference)

    write_json(results_dir / "environment.json", environment)
    write_json(results_dir / "capability_matrix.json", cap_matrix)
    write_task_registry(results_dir / "task_registry.json", selected_specs)
    write_jsonl(results_dir / "results.jsonl", records)
    write_csv(results_dir / "results.csv", records)
    write_report(
        results_dir / "REPORT.md",
        records=records,
        specs=selected_specs,
        environment=environment,
        capability_matrix=cap_matrix,
        h20_reference=h20_reference,
    )

    pass_count = sum(1 for r in records if r.get("validation", {}).get("status") == "pass")
    fail_count = sum(1 for r in records if r.get("validation", {}).get("status") == "fail")
    unsupported_count = sum(1 for r in records if r.get("validation", {}).get("status") == "unsupported")
    print(
        f"[contract-benchmark] records={len(records)} pass={pass_count} "
        f"fail={fail_count} unsupported={unsupported_count}"
    )
    print(f"[contract-benchmark] report: {results_dir / 'REPORT.md'}")
    return 0


def _load_h20_reference(path: Path) -> dict[str, str] | None:
    summary_csv = path / "summary.csv"
    manifest = path / "manifest.json"
    if not summary_csv.exists() and not manifest.exists():
        return None
    ref: dict[str, str] = {}
    if manifest.exists():
        ref["old_manifest"] = str(manifest)
    if summary_csv.exists():
        ref["old_summary"] = str(summary_csv)
    ref["warning"] = "historical old benchmark, different contract; use only for discrepancy analysis"
    return ref


if __name__ == "__main__":
    raise SystemExit(main())
