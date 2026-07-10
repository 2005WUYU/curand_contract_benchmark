# Contract FlagRand vs cuRAND Benchmark

这是按以下两份规划文档重做的新 benchmark：

- `E:\20251018project\internship_flaggem\curand_benchmark_exposure_feasibility_and_refined_plan (1).md`
- `E:\20251018project\internship_flaggem\flagrand_curand_benchmark_logic_design.md`

它独立于旧 `benchmark/` 目录，也不再沿用上一版 `curand_refined_benchmark` 的 MVP 任务集。

## FlagRand 来源

本仓库的 `src/flagrand/` 是重构后
`https://github.com/2005WUYU/FlagRand_RTX4060.git` 在 upstream commit
`14f904077474d276fe9e966f42bb16ad194a1a73` 的 vendored snapshot。
`contract_benchmark/flagrand_vendor.json` 固定 commit 和源码 tree hash；`h20_smoke.sh`、
`h20_hostapi_benchmark.sh`、`h20_benchmark.sh` 都会先运行
`scripts/verify_flagrand_vendor.py`，确认 import 来自本仓库、`flagrand.curand`
facade 完整且 tree hash 一致，再开始 benchmark。

public candidate 统一经重构上游的 `flagrand.curand` facade 调用。Host-API-only
矩阵包含 Philox `uniform_f64`/`normal_f64`/`lognormal_f64`，但 Philox 不支持
`raw64`。完整 task benchmark 中 F0/F1/F2/M2/M3 的 `flagrand_fused_philox` 则来自
`contract_benchmark/kernels.py` 的 benchmark-local Triton 内核，不是上游 FlagRand
fused API；这类结果只能用于 solution-level 对比。

## 本机 smoke

```powershell
& 'E:\conda_envs\gantry\python.exe' `
  'E:\20251018project\internship_flaggem\flagrand-main\curand_contract_benchmark\run_benchmark.py' `
  --profile local_smoke
```

已通过的本机 smoke 结果：

```text
curand_contract_benchmark/results/20260620_171804_local_smoke
records=194 pass=166 fail=0 unsupported=28
```

已通过的本机 local 结果：

```text
curand_contract_benchmark/results/20260620_171937_local
records=419 pass=353 fail=0 unsupported=66
```

`unsupported` 主要是本机未构建 legacy cuRAND Device API extension、未配置 cuRANDDx。它们会显式进入结果，不能被当作性能结论。

## H20 正式运行

H20 登录节点没有 GPU，不要在登录节点裸跑 `python run_benchmark.py`。正式运行必须先用 Slurm 申请计算节点，再在计算节点里启动公司指定 Docker 镜像。

先看完整指南：

```text
docs/H20_CLUSTER_RUN_GUIDE.md
```

推荐顺序是：同 image/node 的 1-GPU smoke → hostapi-only 全量 → 完整 task/gate
benchmark。以下命令假设 `$NODE` 已设为目标 H20 节点。

先跑 smoke：

```bash
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

smoke 达到 `fail=0` 后，跑 hostapi-only 全量：

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

最后跑复杂完整 benchmark：

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
GROUPS=all \
BUILD_DEVICE_EXT=1 \
ALLOW_DEVICE_EXT_FAILURE=0 \
bash scripts/h20_benchmark.sh
```

复杂 benchmark 也可拆成 stage0/1 与 stage2/3/4；完整命令和故障排查见
`docs/H20_CLUSTER_RUN_GUIDE.md`。Host-API-only 的 case matrix 见
`docs/HOSTAPI_ONLY_BENCHMARK.md`。

如果前述 image/node/MathDx 变量已经 `export`，多卡全跑可简写为：

```bash
NUM_GPUS=4 SLURM_PARTITION=long TIME_LIMIT=08:00:00 PROFILE=h20 GROUPS=all BUILD_DEVICE_EXT=1 bash scripts/h20_benchmark.sh
```

这些脚本外层使用 `srun --gres=gpu:<N>`；默认 image/tar 是
`flagtree-nvidia:3.6-v2` 和 `/data/nfs3/flagtree-nvidia-3.6-v2.tar`，上述命令已通过
`IMAGE`/`IMAGE_TAR` 覆盖为带 MathDx/cuRANDDx 的目标镜像。`NUM_GPUS>1` 时会按 task
分片并行跑，然后合并 `results.jsonl`、`results.csv` 和 `REPORT.md`。
带 MathDx/cuRANDDx headers 的镜像中，`scripts/h20_benchmark.sh` 默认也会尝试构建 native cuRANDDx extension；
`ALLOW_DEVICE_EXT_FAILURE=0` 会让 legacy Device API 和 cuRANDDx extension 构建失败都直接终止。
脚本会在容器内自动设置 CUDA runtime `LD_LIBRARY_PATH`，legacy Device API extension 构建脚本也会写入 rpath 并输出 `ldd` 诊断。

## 输出结构

每次运行会生成：

```text
curand_contract_benchmark/results/<timestamp>_<profile>/
  environment.json
  capability_matrix.json
  task_registry.json
  results.jsonl
  results.csv
  summary.json
  REPORT.md
```

正式分析优先看：

- `summary.json`：机器可读汇总，包含 formal speedup 区间、gate failures、unsupported counts。
- `REPORT.md`：人读报告。
- `results.jsonl`：逐行审计，保留 claim、observability class、timing boundary、validation、audit flags 和 unsupported reason。

`run_benchmark.py` 和多卡 launcher 会把 `summary.json.run_health.status != "ok"` 或任意
`validation.status=fail` 视为失败运行并返回非零退出码。`unsupported` 行仍然只是显式能力缺失，
不能当作性能结论。

代码结构说明见：

```text
docs/BENCHMARK_CODE_ARCHITECTURE.md
```

快速 gate smoke 可单独运行：

```bash
bash scripts/gate_smoke.sh
```

## 设计原则

本 benchmark 不输出唯一总分。所有结论必须落到具体任务：

- raw generator
- distribution public API
- setup/state
- call granularity
- fused end-to-end
- QRNG/Sobol
- robustness
- Device API/cuRANDDx support

Fused 结果只能解释完整方案收益，不能解释为 raw RNG 本体更快。
