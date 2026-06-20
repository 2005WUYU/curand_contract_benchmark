# Benchmark Code Architecture

This repository keeps benchmark logic split by responsibility instead of by output file format.

## Core Modules

- `contract_benchmark/profiles.py`
  - Benchmark profiles, run context, environment/provenance capture.
  - This is where CUDA, cuRAND, git, and `nvidia-smi` metadata are collected.

- `contract_benchmark/records.py`
  - Record construction, rough-gate scope tagging, cross-record gates, and speedup computation.
  - Changes here affect what can become a formal result.

- `contract_benchmark/runner.py`
  - Task dispatch and concrete benchmark task implementations.
  - This file should contain benchmark operations, not environment capture or report summarization.

- `contract_benchmark/summary.py`
  - Machine-readable `summary.json` generation from `results.jsonl` records.
  - Use this for downstream analysis instead of reimplementing ad hoc JSONL scans.

- `contract_benchmark/reporting.py`
  - Human-readable Markdown and CSV/report output.

- `contract_benchmark/spec.py`
  - Task registry and task claims.
  - If task semantics change, update this file with the code change.

## Trust Boundaries

- `validation.status == "pass"` means a rough benchmark gate passed, not that a generator passed a full statistical test suite.
- `formal_result == true` means the row passed local validation and cross-record gates and did not carry disqualifying timing audit flags.
- `speedup_gpu_vs_baseline` is only meaningful when `formal_result == true` and `speedup_baseline_formal == true`.
- `unsupported` rows are explicit capability records, not performance failures.

## Debugging Path

1. Start with `summary.json` for counts, formal speedup ranges, failures, and unsupported paths.
2. Use `results.csv` for table inspection.
3. Use `results.jsonl` only when row-level checks or timing details are needed.
4. Use shard logs and `parallel_manifest.json` to distinguish benchmark runtime from final report-write time.

## Pending Larger Changes

- cuRANDDx requires MathDx headers and real kernels, not just a library install.
- Cross-GPU replicate mode should include shard/GPU identity in comparison keys or a separate replicate aggregation path.
- Full random-quality claims should use TestU01, PractRand, or a comparable statistical suite instead of the current rough gates.
