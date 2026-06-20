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

依赖前提：

- Python environment with PyTorch CUDA build.
- Triton.
- CUDA Toolkit / cuRAND shared library available on library path.

```bash
cd /path/to/flagrand-main
python curand_contract_benchmark/run_benchmark.py --profile h20
```

可选强基线：先构建 legacy cuRAND Device API fused extension。

```bash
python curand_contract_benchmark/native/build_curand_device_extension.py --verbose
python curand_contract_benchmark/run_benchmark.py --profile h20 --groups all
```

## 输出结构

每次运行会生成：

```text
curand_contract_benchmark/results/<timestamp>_<profile>/
  environment.json
  capability_matrix.json
  task_registry.json
  results.jsonl
  results.csv
  REPORT.md
```

正式分析优先看 `results.jsonl` 和 `REPORT.md`，因为它们保留了 claim、observability class、timing boundary、validation、raw samples、audit flags 和 unsupported reason。

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
