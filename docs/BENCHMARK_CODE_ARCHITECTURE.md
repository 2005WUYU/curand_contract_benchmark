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
  - Task dispatch only: metadata tasks, capability matrix handling, task_id-to-runner mapping, and final record gates.
  - This file should stay small. Concrete benchmark operations live under `contract_benchmark/tasks/`.

- `contract_benchmark/tasks/`
  - Concrete benchmark implementations split by task family.
  - `gates.py`: G0/G1/G2/G3 rough contract gates.
  - `bulk.py`: H-series host bulk generation and ordering sweep.
  - `lifecycle.py`: I-series lifecycle, seed generation, and first-vs-steady timing.
  - `granularity.py`: A-series call granularity sweeps.
  - `device.py`: K-series device output, M3 device fused consume, and E1 compile support matrix.
  - `fused.py`: F-series fused threshold/add/dropout comparisons.
  - `memory.py`: M0/M1 memory and consume baselines.
  - `qrng.py`: Q-series Sobol tasks.
  - `robustness.py`: E0 host status/error behavior.
  - `common.py`: shared task helpers for validation, common records, legacy-device baselines, and cuRANDDx unsupported rows.

- `contract_benchmark/adapters.py`
  - Compatibility facade for benchmark-facing backend helpers.
  - New code should usually read the focused modules first:
    `generator_registry.py`, `curand_adapter.py`, `flagrand_adapter.py`, and `capabilities.py`.

- `contract_benchmark/curand_ctypes.py`
  - Compatibility facade for the cuRAND Host API binding.
  - ABI details are split into `curand_constants.py`, `curand_library.py`, and `curand_generator.py`.

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
- `backend` is the displayed implementation path, while `gate_backend` is the canonical semantic scope used for cross-record gates. For example, `flagrand_public_output`, `flagrand_public_first`, and `flagrand_public_steady` are gated by `flagrand_public` G1/G2 results.

## Debugging Path

1. Start with `summary.json` for counts, formal speedup ranges, failures, and unsupported paths.
2. Use `results.csv` for table inspection.
3. Use `results.jsonl` only when row-level checks or timing details are needed.
4. Use shard logs and `parallel_manifest.json` to distinguish benchmark runtime from final report-write time.

## Pending Larger Changes

- cuRANDDx requires MathDx headers and real kernels, not just a library install.
- Cross-GPU replicate mode should include shard/GPU identity in comparison keys or a separate replicate aggregation path.
- Full random-quality claims should use TestU01, PractRand, or a comparable statistical suite instead of the current rough gates.
