# Contract FlagRand vs cuRAND Benchmark

这是按以下两份规划文档重做的新 benchmark：

- `E:\20251018project\internship_flaggem\curand_benchmark_exposure_feasibility_and_refined_plan (1).md`
- `E:\20251018project\internship_flaggem\flagrand_curand_benchmark_logic_design.md`

它独立于旧 `benchmark/` 目录，也不再沿用上一版 `curand_refined_benchmark` 的 MVP 任务集。

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

最小 smoke：

```bash
bash scripts/h20_smoke.sh
```

分阶段正式跑：

```bash
SLURM_PARTITION=debug TIME_LIMIT=01:00:00 PROFILE=h20 GROUPS=stage0,stage1 BUILD_DEVICE_EXT=0 bash scripts/h20_benchmark.sh
SLURM_PARTITION=long TIME_LIMIT=08:00:00 PROFILE=h20 GROUPS=stage2,stage3,stage4 BUILD_DEVICE_EXT=1 bash scripts/h20_benchmark.sh
```

一次全跑：

```bash
SLURM_PARTITION=long TIME_LIMIT=08:00:00 PROFILE=h20 GROUPS=all BUILD_DEVICE_EXT=1 bash scripts/h20_benchmark.sh
```

多卡并行全跑：

```bash
NUM_GPUS=4 SLURM_PARTITION=long TIME_LIMIT=08:00:00 PROFILE=h20 GROUPS=all BUILD_DEVICE_EXT=1 bash scripts/h20_benchmark.sh
```

这些脚本外层使用 `srun --gres=gpu:<N>`，内层使用 `flagtree-nvidia:3.6-v2`，镜像缺失时会尝试从 `/data/nfs3/flagtree-nvidia-3.6-v2.tar` 加载。`NUM_GPUS>1` 时会按 task 分片并行跑，然后合并 `results.jsonl`、`results.csv` 和 `REPORT.md`。

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
