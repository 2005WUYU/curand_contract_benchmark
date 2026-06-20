# H20 / 公司集群运行指南

## 1. 目标

这个仓库已经包含两部分：

- `contract_benchmark/`：新的 contract-style cuRAND benchmark。
- `src/flagrand/`：运行 benchmark 必需的 FlagRand 源码与 Sobol/MTGP 参数数据。

因此在公司集群上直接 clone 本仓库即可运行，不需要再额外复制 `flagrand-main/src`。

## 2. 已知旧 H20 环境参考

旧测试目录：

```text
E:\20251018project\internship_flaggem\flagrand-main\20260615-185511-h20-full-fast
```

旧 `environment.json` 记录的关键信息：

```text
GPU: NVIDIA H20
driver: 590.48.01
CUDA_VISIBLE_DEVICES: 0
python: 3.12.3
torch: 2.8.0a0+5228986c39.nv25.5
cwd: /workspace
old timing_mode: api_cuda_event
old stream_mode: continuous_stream
```

旧 summary 的 geomean speedup 主要是：

```text
philox raw: 0.5266
xorwow raw: 0.4404
mrg32k3a raw: 0.5342
uniform public/fused path: 0.2436
normal: 0.2566
lognormal: 0.2604
poisson: 0.3392
```

这些是旧 public API CUDA-event 口径，不要和新 contract benchmark 的任务直接混成同一个结论。

## 3. Clone

```bash
cd /workspace
git clone https://github.com/2005WUYU/curand_contract_benchmark.git
cd curand_contract_benchmark
```

如果公司网络更适合 SSH：

```bash
git clone git@github.com:2005WUYU/curand_contract_benchmark.git
cd curand_contract_benchmark
```

## 4. 环境检查

优先使用公司镜像里已有的 PyTorch CUDA 环境。先检查：

```bash
python - <<'PY'
import torch, sys
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

nvidia-smi
```

如果 `triton` 不存在：

```bash
python -m pip install triton
```

如果要 editable install：

```bash
python -m pip install -e .
```

本 benchmark 的 `run_benchmark.py` 也会把仓库内 `src/` 加入 `sys.path`，所以不 install 也能运行。

## 5. 先跑 smoke

```bash
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --profile local_smoke
```

期望：

```text
fail=0
unsupported 可以存在
```

`unsupported` 常见来源：

- legacy cuRAND Device API extension 尚未构建。
- cuRANDDx 未配置。
- 某些 ordering 当前 generator/config 不支持。

这些不是 smoke 失败，但必须保留在结果里。

## 6. 构建 legacy cuRAND Device API 强基线

如果集群有完整 CUDA Toolkit、NVCC 和 PyTorch extension 构建能力，执行：

```bash
CUDA_VISIBLE_DEVICES=0 python native/build_curand_device_extension.py --verbose
```

构建成功后再跑 fused 任务，结果中会出现：

```text
backend = curand_legacy_device_fused
api_surface = legacy_device_api_extension
```

如果构建失败，不要手动删掉 unsupported 结果；把 build log 和 `unsupported_reason` 放进报告。

## 7. 正式 H20 跑法

建议分三段跑，避免一次全量失败难定位：

```bash
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --profile h20 --groups stage0,stage1
CUDA_VISIBLE_DEVICES=0 python native/build_curand_device_extension.py --verbose
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --profile h20 --groups stage2,stage3,stage4
```

如果希望一次全跑：

```bash
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --profile h20 --groups all
```

结果目录：

```text
results/<timestamp>_h20/
  environment.json
  capability_matrix.json
  task_registry.json
  results.jsonl
  results.csv
  REPORT.md
```

正式分析优先看：

```text
REPORT.md
results.jsonl
capability_matrix.json
```

## 8. 如何判断结果

不要只看一个 speedup。按任务分层看：

```text
H0_RAW32_BULK
  raw generator 本体/写出能力。

H2/H3/H4/H5
  uniform/normal/lognormal/poisson public output 能力。

I1/I2/I3
  lifecycle、GenerateSeeds、first-vs-steady。

A0/A1/A2
  one-call 曲线、小调用固定开销、调用次数放大。

F0/F1/F2
  threshold/add/dropout end-to-end fused 方案收益。

M0/M1/M2/M3
  写出下限、预生成消费、Host bulk+consume、Device/Dx fused 支持情况。

Q0/Q1
  Sobol/QRNG，不能和 PRNG 混成一个平均分。
```

Fused 任务只能说 solution-level：

```text
FlagRand fused 相对 cuRAND Host bulk+consume 快/慢多少。
```

不能说：

```text
FlagRand RNG 本体因此比 cuRAND 快。
```

要回答 RNG 本体，看 `H0` 或 Device output-only 任务。

## 9. 结果回传

建议只打包最终结果目录：

```bash
tar -czf contract_benchmark_h20_results.tar.gz results/<timestamp>_h20
```

如果要保留完整构建信息，也一起保存：

```bash
tar -czf contract_benchmark_h20_debug.tar.gz \
  results/<timestamp>_h20 \
  native/build
```

## 10. 常见问题

### 找不到 `flagrand`

确认当前目录就是 clone 下来的仓库根目录：

```bash
pwd
ls src/flagrand
python - <<'PY'
import sys
sys.path.insert(0, "src")
import flagrand
print(flagrand.__file__)
PY
```

### 找不到 cuRAND shared library

确认 CUDA Toolkit/driver 路径：

```bash
ldconfig -p | grep curand || true
find /usr/local -name 'libcurand.so*' 2>/dev/null | head
```

必要时：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Device extension 构建失败

先跑 Host API 阶段：

```bash
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --profile h20 --groups stage0,stage1
```

Device/Dx 缺失会作为 unsupported 记录，不能把这些行当成性能结论。

