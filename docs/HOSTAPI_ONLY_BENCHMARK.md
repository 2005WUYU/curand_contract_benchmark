# Host API Only Benchmark

This benchmark is a separate path from the contract/task benchmark. It exists for
direct performance comparison between:

- baseline: cuRAND Host API
- candidate: FlagRand public API

It does not run the task registry, contract gates, cuRAND Device API extension,
cuRANDDx extension, fused-consume baselines, or D0 decomposition.

## What It Measures

The benchmark builds a simple case matrix:

```text
generator x distribution x size
```

Each case emits two records:

- `curand_host_api`
- `flagrand_public_api`

For successful pairs, `speedup_gpu_vs_curand_host` is:

```text
cuRAND Host API median CUDA-event time / FlagRand median CUDA-event time
```

Values greater than `1.0` mean FlagRand was faster for that case.

## Default Coverage

For the `h20` profile, the default size list is:

```text
4096, 16384, 65536, 262144, 1048576, 4194304, 8388608
```

Default generators:

```text
philox4x32_10, xorwow, mrg32k3a, mtgp32, mt19937,
sobol32, scrambled_sobol32, sobol64, scrambled_sobol64
```

Default distributions:

```text
raw, uniform, normal, lognormal, poisson
```

Distribution expansion is generator-aware:

- 32-bit generators use `raw32`, `uniform_f32`, `normal_f32`, `lognormal_f32`.
- 64-bit Sobol generators use `raw64`, `uniform_f64`, `normal_f64`, `lognormal_f64`.
- Poisson is limited to 32-bit PRNG generators.
- FlagRand large-lambda Poisson rows are marked as approximation-path rows in case notes.

## H20 Run

Use the same node-local image setup as the full benchmark, but call the hostapi
entry point:

```bash
cd ~/workspace/curand_contract_benchmark
git pull --ff-only origin main

IMAGE=flagrand-cuda13-curanddx:latest \
H20_NODELIST=bjdb-h20-node-038 \
MATHDX_ROOT=/opt/mathdx/current \
CPATH=/opt/mathdx/current/include/curanddx:/opt/mathdx/current/include \
CMAKE_PREFIX_PATH=/opt/mathdx/current \
NUM_GPUS=4 \
SLURM_PARTITION=long \
TIME_LIMIT=08:00:00 \
PROFILE=h20 \
bash scripts/h20_hostapi_benchmark.sh
```

The launcher still uses `scripts/h20_srun_docker.sh`, so node-local image
selection and result-spool fallback behavior are shared with the full benchmark.

## Useful Narrow Runs

Focus on the generators discussed most often:

```bash
HOSTAPI_GENERATORS=philox4x32_10,mtgp32,sobol32,sobol64 \
bash scripts/h20_hostapi_benchmark.sh
```

Run only raw and uniform:

```bash
HOSTAPI_DISTRIBUTIONS=raw,uniform \
bash scripts/h20_hostapi_benchmark.sh
```

Run a small debug slice:

```bash
HOSTAPI_GENERATORS=philox4x32_10 \
HOSTAPI_DISTRIBUTIONS=raw,uniform \
HOSTAPI_SIZES=4096,1048576 \
HOSTAPI_REPEATS=3 \
bash scripts/h20_hostapi_benchmark.sh
```

## Outputs

The run directory is under:

```text
results/hostapi_only/
```

Important files:

- `REPORT.md`: human-readable summary.
- `records.csv`: compact table for analysis.
- `records.jsonl`: full records, including raw repeat samples.
- `summary.json`: status counts and speedup summary.
- `case_matrix.json`: generated case matrix.
- `environment.json`: CUDA, torch, cuRAND, git, and runtime-cache metadata.

For multi-GPU runs, the root directory also includes shard logs and
`parallel_manifest.json`.

