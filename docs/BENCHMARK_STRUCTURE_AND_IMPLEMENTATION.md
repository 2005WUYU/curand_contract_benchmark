# cuRAND Contract Benchmark 结构与实现说明

## 1. 为什么重做

这版 benchmark 把两份规划文档当作规格书，而不是把它们当背景说明。核心变化是：

- 先写 claim，再写 task。
- 每个 task 必须有 output contract、timing boundary、validation gate。
- cuRAND 被拆成 Host API、legacy Device API、cuRANDDx 三类可见边界。
- public-output、setup、call-granularity、fused solution、QRNG、robustness 分开记录。
- unsupported baseline 显式入结果，不允许静默缺失。

旧 H20 benchmark 使用的是 `api_cuda_event` 包 public API 的粗粒度口径，它可以作为历史参照，但不能替代这版 contract benchmark。

## 2. 目录结构

```text
curand_contract_benchmark/
  run_benchmark.py
  README.md
  contract_benchmark/
    spec.py
    runner.py
    adapters.py
    curand_ctypes.py
    timing.py
    validation.py
    kernels.py
    reporting.py
    optional_device_api.py
  native/
    curand_contract_device_ext.cu
    build_curand_device_extension.py
  docs/
    BENCHMARK_STRUCTURE_AND_IMPLEMENTATION.md
```

`spec.py` 是规格注册表。它定义 C0/C1/G0/G1/G2/G3/H0-H6/I1-I3/A0-A2/K0/K1/F0-F2/M0-M3/Q0/Q1/E0/E1，每个任务都带 claim、comparison level、observability class、timing boundary、validation gate、允许结论和禁止结论。

`runner.py` 是执行层。它不重新解释结果，只按 spec 执行并记录。

`curand_ctypes.py` 直接加载 cuRAND Host API，覆盖 raw32、Sobol64 raw64、uniform、normal、lognormal、Poisson、ordering、offset、dimensions、GenerateSeeds。

`timing.py` 同时保存 CUDA event、CPU enqueue、wall-sync 三种数据。Host API 异步调用必须用同 stream event 或明确同步后的 wall time。

`validation.py` 只做 performance gate 粗检，不声称完整随机质量证明。

`kernels.py` 提供消费 kernel 和 fused Philox/Triton kernel，用于 F0/F1/F2/M0/M1 控制任务。

`native/` 提供 legacy cuRAND Device API fused extension。Windows 本机未必能构建；H20/Linux 上可构建后作为设备端 fused baseline。

## 3. 任务体系

### Stage 0：语义和能力固定

- `C0_CAPABILITY_MATRIX`
- `C1_VERSION_SYMBOL_SELFTEST`

目的：先确认哪些 Host/Device/Dx/FlagRand 路径真实可用。不可用的路径必须记录 `unsupported_reason`。

### Stage 1：Host API 直接可测闭环

- `G0_BASIC_CONTRACT`
- `G1_DISTRIBUTION_ROUGH_CHECK`
- `G2_REPRODUCIBILITY`
- `G3_SEQUENCE_COUNTER_BUDGET`
- `H0_RAW32_BULK`
- `H1_RAW64_SOBOL_BULK`
- `H2_UNIFORM_F32_BULK`
- `H3_NORMAL_F32_BULK`
- `H4_LOGNORMAL_F32_BULK`
- `H5_POISSON_LAMBDA_SWEEP`
- `H6_ORDERING_SWEEP`
- `I1_GENERATOR_LIFECYCLE`
- `I2_CURAND_GENERATE_SEEDS`
- `I3_FIRST_VS_STEADY`
- `A0_SINGLE_CALL_CURVE`
- `A1_FIXED_TOTAL_MANY_SMALL`
- `A2_FIXED_CHUNK_CALLS_SWEEP`
- `Q0_RAW_SOBOL`
- `Q1_SOBOL_D_DIM_UNIT_CUBE`
- `E0_HOST_STATUS_MATRIX`

目的：形成 Host API 公开边界下的完整闭环，区分 raw、distribution、setup、small-call、QRNG 和错误处理。

### Stage 2：设备端基线

- `K0_DEVICE_RAW_OUTPUT`
- `K1_DEVICE_UNIFORM_OUTPUT`
- `E1_COMPILE_SUPPORT_MATRIX`

目的：legacy Device API/cuRANDDx 可用时作为更强基线；不可用时显式 unsupported。

### Stage 3：融合完整任务

- `F0_THRESHOLD_BERNOULLI`
- `F1_ADD_UNIFORM`
- `F2_DROPOUT`
- `M2_HOST_BULK_CONSUME`
- `M3_DEVICE_DX_FUSED_CONSUME`

目的：比较随机数立即消费的完整方案。这里的 speedup 是 solution-level，不是 raw RNG-level。

### Stage 4：解释控制实验

- `M0_PURE_WRITE`
- `M1_PREGENERATED_CONSUME`

目的：帮助解释 F 类任务里内存写出、读入和消费 kernel 的下限成本。

## 4. 结果记录

每条记录至少包含：

- `task_id`
- `claim_id`
- `comparison_level`
- `observability_class`
- `backend`
- `api_surface`
- `generator`
- `distribution`
- `ordering`
- `seed`
- `offset`
- `N`
- `parameters`
- `timing_boundary`
- `validation`
- `raw_samples_us`
- `wall_sync_samples_us`
- `cpu_enqueue_samples_us`
- `median_gpu_us`
- `median_wall_sync_us`
- `speedup_gpu_vs_baseline`
- `speedup_wall_vs_baseline`
- `audit_flags`
- `formal_result`
- `what_it_can_say`
- `what_it_cannot_say`

`formal_result=false` 的记录可以用于诊断，但不应写成正式性能结论。

## 5. F1 单算子如何解释

F1 的任务是：

```text
y = x + alpha * (u - 0.5)
```

cuRAND Host baseline：

```text
curandGenerateUniform -> temporary u buffer -> consume_add_uniform kernel
```

FlagRand fused candidate：

```text
one fused Philox/Triton kernel -> generate u in registers -> write y
```

如果 F1 中 FlagRand 更快，允许说：

```text
在 F1 end-to-end solution 任务中，FlagRand fused 减少了临时随机数组写出/读回和一次消费 kernel 调用，因此完整方案更快。
```

不允许说：

```text
F1 更快，所以 FlagRand raw RNG 本体更快。
```

要判断 raw RNG 本体，必须看 `H0_RAW32_BULK` 或设备端 output-only baseline。

## 6. 本机 smoke 校验

最近一次本机 smoke：

```text
E:\20251018project\internship_flaggem\flagrand-main\curand_contract_benchmark\results\20260620_171804_local_smoke
records=194
pass=166
fail=0
unsupported=28
```

最近一次本机 local：

```text
E:\20251018project\internship_flaggem\flagrand-main\curand_contract_benchmark\results\20260620_171937_local
records=419
pass=353
fail=0
unsupported=66
```

`unsupported` 不是失败，主要来自：

- legacy cuRAND Device API extension 未在本机构建。
- cuRANDDx 未配置。
- FlagRand 没有 `curandGenerateSeeds` 的同级 Host API 概念。
- 某些 cuRAND ordering 不支持当前 generator/config。

## 7. 和旧 H20 benchmark 的差别

旧 H20 目录：

```text
E:\20251018project\internship_flaggem\flagrand-main\20260615-185511-h20-full-fast
```

旧结果的 manifest 显示：

```text
timing_mode = api_cuda_event
stream_mode = continuous_stream
```

旧 summary 中 geomean speedup 典型值：

```text
philox raw: 0.5266
xorwow raw: 0.4404
mrg32k3a raw: 0.5342
uniform fused/public path: 0.2436
normal: 0.2566
lognormal: 0.2604
poisson: 0.3392
```

这些数字说明在旧 public API CUDA-event 口径下，FlagRand 大多慢于 cuRAND。新版 benchmark 不会拿本机 RTX 4060 的 smoke 数字去推翻这个结论；新版要做的是把“为什么慢、慢在哪一层、fused 是否只是相对 Host bulk+consume 有收益、Device/Dx 强基线是否仍成立”拆开。

## 8. H20 正式分析建议

H20 上建议按顺序执行：

```bash
python curand_contract_benchmark/run_benchmark.py --profile h20 --groups stage0,stage1
python curand_contract_benchmark/native/build_curand_device_extension.py --verbose
python curand_contract_benchmark/run_benchmark.py --profile h20 --groups stage2,stage3,stage4
```

正式报告应优先回答：

- `H0` raw bulk 是否仍落后。
- `H2/H3/H4/H5` distribution public API 落后是否大于 raw。
- `I1/I2/I3` setup 是否是瓶颈。
- `A1/A2` 小调用是否显著恶化。
- `F0/F1/F2` 相对 Host bulk+consume 是否有 solution-level 收益。
- legacy Device fused/cuRANDDx 可用后，FlagRand fused 是否仍有优势。
