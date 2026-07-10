# Host API Only Benchmark

This benchmark is a separate path from the contract/task benchmark. It exists for
direct performance comparison between:

- baseline: cuRAND Host API
- candidate: FlagRand public API

It does not run the task registry, contract gates, cuRAND Device API extension,
cuRANDDx extension, fused-consume baselines, or D0 decomposition.

## FlagRand Source and API Boundary

The candidate is the vendored snapshot under `src/flagrand/`, taken from:

```text
repository: https://github.com/2005WUYU/FlagRand_RTX4060.git
upstream commit: 14f904077474d276fe9e966f42bb16ad194a1a73
API surface: flagrand.curand
```

`contract_benchmark/flagrand_vendor.json` pins this commit and the expected hash of the vendored
`.py`/`.pt` tree. `scripts/h20_hostapi_benchmark.sh` runs
`scripts/verify_flagrand_vendor.py` inside the benchmark container before any
case starts. The preflight fails the job if `flagrand` resolves outside this
repository, the `flagrand.curand` facade is missing required calls, or the
vendored tree differs from the manifest. It verifies the checked-out snapshot;
it does not fetch or update FlagRand.

The benchmark-side adapter calls the refactored upstream `flagrand.curand`
facade (`create_generator` and the dtype-specific `generate*` calls).
This facade presents a cuRAND-shaped Python API over FlagRand generators; it is
not a wrapper around `libcurand` and does not imply cuRAND stream/order
equivalence. The baseline remains the real cuRAND Host API.
Each case creates/configures its generator before validation and timing; the
reported generation timings cover repeated `generate*` calls, not generator
construction. Lifecycle cost remains a separate task in the full benchmark.

The host-API-only entry point does not use the task benchmark's
`contract_benchmark/kernels.py`. In particular, records named
`flagrand_fused_philox` in F0/F1/F2/M2/M3 belong to benchmark-local Triton experiment
kernels, not to the upstream FlagRand package. They must not be cited as an
upstream FlagRand fused API result. This is separate from the implementation
under `src/flagrand/fused/`, which is part of the pinned upstream snapshot and
is reached by the public `flagrand.curand` facade.

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

Every successful record now carries three complementary timing views:

- single-call CUDA event, wall-sync, and CPU enqueue medians;
- a batched CUDA-event steady-state value reported in microseconds per call;
- `GPU event - CPU enqueue`, explicitly labeled as a diagnostic subtraction of
  independent medians rather than a formal kernel time.

The batched view uses at most 32 calls and caps each sample near 16M generated
items (`max(1, min(32, 16777216 // N))` by default). This keeps large-N runs
bounded while giving small calls enough queued work to expose steady-state
stream throughput. CUDA events still include any GPU idle gaps caused by slow
host submission, so the batched number must not be described as pure kernel
execution time.

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

- Philox uses `raw32` and both f32/f64 distribution rows:
  `uniform_f32`, `uniform_f64`, `normal_f32`, `normal_f64`,
  `lognormal_f32`, and `lognormal_f64`.
- Philox f64 distribution rows use the pinned upstream direct-Philox f64 path.
  They require an even output count and a Philox4-aligned offset. Generated
  case sizes are aligned automatically; a custom `HOSTAPI_OFFSET` must still
  be a multiple of four.
- Philox does **not** expose native `raw64`; do not generalize its f64
  distribution coverage into a raw64 claim.
- Other 32-bit generators use `raw32`, `uniform_f32`, `normal_f32`,
  `lognormal_f32` where supported.
- 64-bit Sobol generators use `raw64`, `uniform_f64`, `normal_f64`, `lognormal_f64`.
- Custom `HOSTAPI_QRNG_DIMENSIONS` must be between 1 and 20,000. Case sizes are aligned
  to a multiple of the requested dimensions (and any distribution alignment),
  so both backends receive the same valid element count.
- A non-zero `HOSTAPI_OFFSET` is accepted only when every selected generator
  has a comparable offset contract; Philox offsets must also be multiples of
  four. Mixed default-generator runs should keep `HOSTAPI_OFFSET=0`.
- Poisson is limited to 32-bit PRNG generators.
- FlagRand large-lambda Poisson rows are marked as approximation-path rows in case notes.

## H20 Run

After pulling the target commit, first run the repository smoke on one GPU. Use
the same image and node that will be used for the formal run so the smoke also
checks the vendored-source preflight in the real container:

```bash
cd ~/workspace/curand_contract_benchmark
git pull --ff-only origin main

IMAGE=flagrand-cuda13-curanddx:latest \
IMAGE_TAR=/data/nfs3/flagrand-cuda13-curanddx-latest.tar \
H20_NODELIST=$NODE \
MATHDX_ROOT=/opt/mathdx/current \
CPATH=/opt/mathdx/current/include/curanddx:/opt/mathdx/current/include \
CMAKE_PREFIX_PATH=/opt/mathdx/current \
NUM_GPUS=1 \
MEM_PER_GPU_MB=32768 \
SLURM_PARTITION=debug \
TIME_LIMIT=01:00:00 \
bash scripts/h20_smoke.sh
```

Only after smoke reports `fail=0`, run the full host-API-only matrix:

```bash
IMAGE=flagrand-cuda13-curanddx:latest \
IMAGE_TAR=/data/nfs3/flagrand-cuda13-curanddx-latest.tar \
H20_NODELIST=$NODE \
MATHDX_ROOT=/opt/mathdx/current \
CPATH=/opt/mathdx/current/include/curanddx:/opt/mathdx/current/include \
CMAKE_PREFIX_PATH=/opt/mathdx/current \
NUM_GPUS=4 \
MEM_PER_GPU_MB=200000 \
SLURM_PARTITION=long \
TIME_LIMIT=08:00:00 \
PROFILE=h20 \
bash scripts/h20_hostapi_benchmark.sh
```

The launcher still uses `scripts/h20_srun_docker.sh`, so node-local image
selection and result-spool fallback behavior are shared with the full benchmark.
The formal launcher runs the same vendored-source preflight again; a stale,
partially copied, or externally imported FlagRand tree terminates the run before
timings are recorded.

Any runtime exception, validation failure, or unsupported row makes
`run_health=needs_attention` and the launcher exits non-zero. A completed Slurm
process is therefore not treated as a successful benchmark when the refactored
API or a Triton kernel failed inside a case.

## RTX Local Gate

Use `rtx4060_gate` on the actual RTX4060 test machine for an exactly comparable
case matrix. The current Mac development machine is only used to edit, validate
the matrix, and push the code; it cannot execute the CUDA timing run. The profile
has the same generators, distribution expansion, seven sizes, nine Poisson
lambdas, warmup, and repeat counts as `h20`, producing 588 cases with the
default selectors:

```bash
PROFILE=rtx4060_gate \
NUM_GPUS=1 \
python scripts/hostapi_only_benchmark.py
```

Before a long run, verify matrix parity without requiring CUDA execution:

```bash
python scripts/hostapi_only_benchmark.py --profile rtx4060_gate --list-cases > rtx4060_gate_cases.jsonl
python scripts/hostapi_only_benchmark.py --profile h20 --list-cases > h20_cases.jsonl
```

The two files should be byte-for-byte identical. Use the full contract
benchmark separately for fused threshold/add/dropout and bulk-consume task
results; `hostapi_only` deliberately does not fold fused-task performance into
its headline.

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
- `environment.json`: CUDA, torch, cuRAND, git, runtime-cache, and vendored
  FlagRand provenance metadata (including the pinned upstream commit and tree
  verification result).

For multi-GPU runs, the root directory also includes shard logs and
`parallel_manifest.json`.

`summary.json` and `REPORT.md` split the paired result by `N`, generator, and
FlagRand `path_kind`. They report single-event, wall, enqueue, batched-event,
and diagnostic residual speedups side by side so a submission-bound aggregate
cannot be mistaken for a broad kernel regression.
