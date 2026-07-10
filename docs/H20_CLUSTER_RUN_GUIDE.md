# H20 / 公司集群运行指南

## 1. 先说清楚运行边界

H20 集群登录节点没有 GPU，`nvidia-smi` 找不到是正常的。不要在登录节点裸跑 benchmark，也不要在登录节点手动折腾 Python CUDA 环境。

正确外层是：

```text
登录节点 clone / 编辑 / 发起任务
  -> srun 申请 H20 计算节点 GPU
    -> 计算节点内 docker run 项目指定镜像
      -> 容器内部执行 python run_benchmark.py
```

本仓库已经包含：

- `contract_benchmark/`：新的 contract-style cuRAND benchmark。
- `src/flagrand/`：运行 benchmark 必需的重构后 FlagRand 源码与 Sobol/MTGP 参数数据。
- `scripts/h20_*.sh`：按公司 Slurm + Docker 规范封装的运行脚本。

因此公司集群上 clone 本仓库即可运行，不需要再额外复制 `flagrand-main/src`。

### FlagRand 来源与边界

当前 vendored 源码固定为：

```text
repository: https://github.com/2005WUYU/FlagRand_RTX4060.git
upstream commit: 14f904077474d276fe9e966f42bb16ad194a1a73
source: src/flagrand
public compatibility facade: flagrand.curand
```

`contract_benchmark/flagrand_vendor.json` 同时记录 upstream commit、文件数和 `.py`/`.pt` tree hash。
`scripts/verify_flagrand_vendor.py` 会验证实际 import 来自本仓库的 `src/flagrand`、
`flagrand.curand` 必需符号完整，并且源码树与 manifest 一致。它只做验证，不会联网 fetch、
checkout 或静默替换源码。

本 benchmark 的 FlagRand public API 路径统一通过重构上游提供的 `flagrand.curand` facade。
这个 facade 是 FlagRand generator 上的 cuRAND-shaped Python API，不是 `libcurand` wrapper，
也不表示输出 ordering 与 cuRAND 相同。

Host-API-only 矩阵中，Philox 除 f32 外还覆盖 upstream direct path 的
`uniform_f64`、`normal_f64`、`lognormal_f64`；这不等于 Philox 支持 `raw64`。
Philox 原始输出仍只有 `raw32`，native `raw64` 只属于 Sobol64 families。

还要区分两类名字里都含 “fused” 的代码：

- `src/flagrand/fused/` 属于上述固定 commit 的上游 FlagRand 源码，public facade 会调用它。
- `contract_benchmark/kernels.py` 中 F0/F1/F2/M2/M3 的 Philox generate+consume 内核属于
  benchmark-local 实验实现；结果虽使用 `flagrand_fused_philox` 名称，但 record 的
  `api_surface` 是 `flagrand_benchmark_kernel`，不能写成“上游 FlagRand fused API”。

## 2. 来自公司规范的关键约束

参考本地规范目录：

```text
E:\20251018project\internship_flaggem\公司集群使用规范
```

本 benchmark 用到的关键文件：

- `H20-slurm使用规范.md`
- `第一阶段跑通步骤.md`
- `grun使用指南.md`
- `gpu-check使用文档.md`

必须遵守的点：

- H20 只有计算节点有 GPU，必须通过 Slurm 排队使用。
- `srun` 不要用 `--gpus-per-task`，使用 `--gres=gpu:<N>`。
- `debug` 队列优先级高但限时 1 小时；正式长跑用 `long`。
- 公司第一阶段文档指定使用 Docker 镜像 `flagtree-nvidia:3.6-v2`。
- 计算节点第一次没有镜像时，从 `/data/nfs3/flagtree-nvidia-3.6-v2.tar` 加载。
- Docker 异常退出时要确认容器是否残留；本仓库脚本使用 `docker run --rm`，但异常中断后仍建议检查。
- `grun` 是不支持 `srun` 的开发机上的用卡工具，不是 H20 Slurm 正式跑法。

## 3. 登录节点准备

```bash
sinfo
sinfo -N -h -O NodeHost:20,Gres:30,GresUsed:30 | sort -u
```

看到 `debug` 或 `long` 队列里有可用节点后，clone 仓库：

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/2005WUYU/curand_contract_benchmark.git
cd curand_contract_benchmark
```

如果公司网络或账号更适合 SSH：

```bash
git clone git@github.com:2005WUYU/curand_contract_benchmark.git
cd curand_contract_benchmark
```

## 4. GPU 与 Docker 预检

先只申请 1 张 H20 做最小验证：

```bash
srun -p debug --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:10:00 bash -lc '
  hostname
  nvidia-smi
'
```

再验证 Docker 镜像：

```bash
srun -p debug --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 bash -lc '
  docker image inspect flagtree-nvidia:3.6-v2 >/dev/null 2>&1 || docker load -i /data/nfs3/flagtree-nvidia-3.6-v2.tar
  docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS:-0} \
    flagtree-nvidia:3.6-v2 \
    bash -lc "python - <<PY
import sys, torch, triton
print(sys.version)
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)
print(\"triton\", triton.__version__)
print(\"cuda available\", torch.cuda.is_available())
print(\"device\", torch.cuda.get_device_name(0))
PY"
'
```

## 5. 推荐：直接用仓库脚本跑

### 5.1 Smoke

在仓库根目录执行。以下命令假设 `$NODE` 已设为正式测试所用、且本地已有目标镜像的
H20 节点；smoke 与正式跑必须使用同一个 image/node 组合：

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

它实际做的是：

```text
srun -p debug --gres=gpu:1 ...
  docker run --rm --gpus all -v "$PWD":/workspace -w /workspace <IMAGE>
    python scripts/verify_flagrand_vendor.py
      && python run_benchmark.py --profile local_smoke
```

期望：

```text
[flagrand-preflight] ... verified=true
fail=0
unsupported 可以存在
```

任何 preflight 错误都应直接停止：这通常表示 H20 拉取的仓库版本不完整、vendored tree
被改动，或容器从别处 import 了同名 `flagrand`。不要在 preflight 失败时继续收集 timing。

`unsupported` 常见来源：

- legacy cuRAND Device API extension 尚未构建。
- cuRANDDx headers 未出现在实际 Docker 容器环境中，或 headers 已存在但 benchmark 尚无 cuRANDDx timing extension。
- 某些 ordering 当前 generator/config 不支持。

这些不是 smoke 失败，但必须保留在结果里。

登录节点只负责提交任务和挂载仓库；真实 Python/CUDA/cuRANDDx 环境以
`docker run ... <image>` 内部为准。

`capability_matrix.json` 和 `REPORT.md` 会记录 cuRANDDx headers 是否在容器内被找到、
cuRANDDx extension 是否成功 import、构建目录和导出的符号。
Docker 镜像是节点本地资源；如果镜像只在特定节点上存在，需要设置
`H20_NODELIST`/`SLURM_NODELIST`，或者设置 `IMAGE_TAR` 为真正包含该镜像 tag 的 tar 包。
legacy cuRAND Device API extension 默认构建到容器内 `/tmp/curand_contract_device_ext_$USER`；
如需覆盖可设置 `CURAND_CONTRACT_DEVICE_BUILD_DIR`，不要依赖挂载仓库内的 `native/build` 可写。
脚本会在容器内自动把 `/usr/local/cuda*`、`/opt/conda/lib` 和 Python wheel 下的
NVIDIA runtime library 目录加入 `LD_LIBRARY_PATH`，并且 legacy Device API extension
构建脚本会把这些路径写入 rpath。`capability_matrix.json`/`REPORT.md` 会记录
`cudart_candidates`、`shared_objects` 和 `missing_dependencies`，用于定位
`libcudart.so.*` 这类动态依赖问题。
cuRANDDx extension 默认构建到容器内 `/tmp/curand_contract_curanddx_ext_$USER`；
如需覆盖可设置 `CURAND_CONTRACT_CURANDDX_BUILD_DIR`。它需要 MathDx/cuRANDDx headers，
通常由 `MATHDX_ROOT=/opt/mathdx/current` 和对应 `CPATH` 提供。
launcher 会用提交任务用户的 UID/GID 运行容器，避免 NFS/root-squash 环境下无法写
`/workspace/results`。
如果容器仍无法直接写仓库挂载，launcher 会把 `/workspace/results` 单独挂到计算节点
`/tmp` spool 目录，Docker 结束后再由外层 Slurm shell 复制回仓库 `results/`。
如果仓库 `results/` 本身不可写，launcher 会自动退到 `h20_results/`，再退到
`$HOME/curand_contract_benchmark_results`；也可以显式设置 `H20_RESULTS_TARGET`。

### 5.2 分阶段正式跑

推荐顺序固定为：

```text
同 image/node 的 1-GPU smoke
  -> hostapi-only 全量（先验证最直接的 public API 对比）
    -> 完整 task/gate benchmark（含 Device/cuRANDDx extension 与 fused/memory/QRNG）
```

第 1 步 smoke 必须先达到 `fail=0`。第 2 步只对比 cuRAND Host API 与 FlagRand
public API，不使用原 task/gate benchmark；执行：

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

hostapi-only 中任何 runtime error、validation failure 或 unsupported row 都会把
`run_health` 标为 `needs_attention` 并让 launcher 非零退出，不能只看 Slurm job 是否跑完。

详细范围、参数和输出见 `docs/HOSTAPI_ONLY_BENCHMARK.md`。

第 3 步运行复杂完整 benchmark：

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

两个正式 launcher 都会在计时前重新运行 vendored FlagRand preflight。复杂 benchmark
的 preflight 位于 native Device/cuRANDDx extension 构建之前，因此源码来源不对时会尽早退出。

如果需要把复杂 benchmark 拆阶段跑，使用下面两步代替上面的 `GROUPS=all`。

先跑 Host API 与基础任务，适合 `debug` 队列：

```bash
SLURM_PARTITION=debug \
TIME_LIMIT=01:00:00 \
PROFILE=h20 \
GROUPS=stage0,stage1 \
BUILD_DEVICE_EXT=0 \
bash scripts/h20_benchmark.sh
```

再构建 legacy cuRAND Device API extension，并跑 fused / memory / QRNG 等后续任务。建议用 `long` 队列：

```bash
SLURM_PARTITION=long \
TIME_LIMIT=08:00:00 \
PROFILE=h20 \
GROUPS=stage2,stage3,stage4 \
BUILD_DEVICE_EXT=1 \
bash scripts/h20_benchmark.sh
```

如果已经在 shell 中 `export` 了前述 image/node/MathDx 变量，也可用简写一次全跑：

```bash
SLURM_PARTITION=long \
TIME_LIMIT=08:00:00 \
PROFILE=h20 \
GROUPS=all \
BUILD_DEVICE_EXT=1 \
bash scripts/h20_benchmark.sh
```

如果要多占几张 GPU 并行切分任务，加 `NUM_GPUS`。例如 4 卡全量：

```bash
NUM_GPUS=4 \
SLURM_PARTITION=long \
TIME_LIMIT=08:00:00 \
PROFILE=h20 \
GROUPS=all \
BUILD_DEVICE_EXT=1 \
bash scripts/h20_benchmark.sh
```

`NUM_GPUS=1` 时输出普通单进程目录；`NUM_GPUS>1` 时脚本会在容器内按 task 分片，每张 GPU 一个进程，最后合并到：

```text
results/<timestamp>_h20_parallel_<N>gpu/
  parallel_manifest.json
  summary.json
  shard_*.log
  shard_*_gpu_*/
  results.jsonl
  results.csv
  REPORT.md
```

脚本会在容器里设置可写缓存目录，避免 Triton fallback 到 `/.triton`：

```text
HOME=/tmp/curand_contract_home_$USER
XDG_CACHE_HOME=/tmp/curand_contract_cache_$USER
TRITON_CACHE_DIR=/tmp/curand_contract_cache_$USER/triton
TORCH_EXTENSIONS_DIR=/tmp/curand_contract_cache_$USER/torch_extensions
```

多卡并行时，每个 shard 会使用独立的 `TRITON_CACHE_DIR`。如果任何 shard 进程返回非零，
launcher 会尽量聚合已有输出、写出 `summary.json` 和 `REPORT.md`，然后整体返回非零。

注意：不要只在 Slurm 层多申请 GPU 却仍单进程跑，那只会多占卡不会变快。本脚本在 `NUM_GPUS>1` 时会自动并行分片。

### 5.3 可调参数

脚本支持这些环境变量：

```text
SLURM_PARTITION   默认 smoke=debug，benchmark=long
TIME_LIMIT        默认 smoke=01:00:00，benchmark=08:00:00
NUM_GPUS          默认 1
CPUS_PER_GPU      smoke 默认 8，benchmark 默认 24
MEM_PER_GPU_MB    smoke 默认 32768，benchmark 默认 242144
IMAGE             默认 flagtree-nvidia:3.6-v2
IMAGE_TAR         默认 /data/nfs3/flagtree-nvidia-3.6-v2.tar
H20_NODELIST      可选，指定已有目标镜像/环境的 H20 节点
PROFILE           默认 h20
GROUPS            默认 all；也支持 BENCHMARK_GROUPS，避免和 Bash 特殊变量重名
BUILD_DEVICE_EXT  1 表示先尝试构建 native cuRAND Device API extension
ALLOW_DEVICE_EXT_FAILURE  默认 1；设为 0 时 Device API extension 构建失败会直接终止
BUILD_CURANDDX_EXT  默认 1，表示先尝试构建 native cuRANDDx extension
ALLOW_CURANDDX_EXT_FAILURE  默认跟随 ALLOW_DEVICE_EXT_FAILURE；设为 0 时 cuRANDDx extension 构建失败会直接终止
MATHDX_ROOT / CPATH / CMAKE_PREFIX_PATH  传给容器内 cuRANDDx header 检测与后续构建
CONTAINER_XDG_CACHE_HOME / CONTAINER_TRITON_CACHE_DIR  可选，覆盖容器内缓存目录
```

## 6. 手写 srun 命令时的等价写法

如果不用脚本，外层也必须是 Slurm + Docker。下面是 smoke 的等价命令：

```bash
srun -p debug --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 bash -lc '
  docker image inspect flagtree-nvidia:3.6-v2 >/dev/null 2>&1 || docker load -i /data/nfs3/flagtree-nvidia-3.6-v2.tar
  docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS:-0} \
    -v "$PWD":/workspace \
    -w /workspace \
    flagtree-nvidia:3.6-v2 \
    bash -lc "python scripts/verify_flagrand_vendor.py && python run_benchmark.py --profile local_smoke"
'
```

正式 H20 不建议全手写，优先用 `scripts/h20_benchmark.sh`，避免漏掉资源、镜像、挂载、工作目录和 `CUDA_VISIBLE_DEVICES`。

## 7. 输出结构

结果目录：

```text
results/<timestamp>_<profile>/
  environment.json
  capability_matrix.json
  summary.json
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

只要 `summary.json` 中 `run_health.status` 不是 `ok`，或 `status_counts.fail` 非 0，
这轮就不能作为成功 benchmark 使用。

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

此外，F0/F1/F2/M2/M3 的 `flagrand_fused_philox` 来自
`contract_benchmark/kernels.py` 的 benchmark-local Triton 内核，并非 pinned upstream
FlagRand 的 `src/flagrand/fused/` public implementation。报告中应写“benchmark-local
FlagRand-side fused candidate”或同等明确的表述，不能把它当成 upstream FlagRand 产品能力。

## 9. 旧 H20 benchmark 参考

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

## 10. 结果回传

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

## 11. 常见问题

### 登录节点 `nvidia-smi` 失败

正常。登录节点没有 GPU。用 `srun` 申请计算节点后再测。

### Docker 权限失败

当前用户可能没有 Docker 权限，需要找运维或 mentor 开权限。

### 镜像不存在

脚本会自动尝试：

```bash
docker load -i /data/nfs3/flagtree-nvidia-3.6-v2.tar
```

如果这个路径不可读，换节点或找 mentor 确认镜像路径。

### 找不到 `flagrand`

确认容器内工作目录是 `/workspace`，并且挂载的是本仓库根目录：

```bash
ls src/flagrand
python - <<'PY'
import sys
sys.path.insert(0, "src")
import flagrand
print(flagrand.__file__)
PY
```

### FlagRand preflight 失败

先直接复现验证，并检查输出中的 expected/module path、缺失 facade symbol 和 tree hash：

```bash
python scripts/verify_flagrand_vendor.py
git status --short src/flagrand contract_benchmark/flagrand_vendor.json
git rev-parse HEAD
```

preflight 校验的是 vendored upstream snapshot，不要求 benchmark 仓库自己的 commit 等于
FlagRand upstream commit。正确状态是 `contract_benchmark/flagrand_vendor.json` 内记录的 upstream commit 为
`14f904077474d276fe9e966f42bb16ad194a1a73`，并且输出 `verified=true`。如果 tree hash 不符，
应回到本机确认 vendoring 后重新 commit/push，再在 H20 端 `git pull --ff-only`；不要在 H20
上临时改 manifest 绕过检查。

### 找不到 cuRAND shared library

确认 CUDA Toolkit/driver 路径：

```bash
ldconfig -p | grep curand || true
find /usr/local -name 'libcurand.so*' 2>/dev/null | head
```

`scripts/h20_benchmark.sh` 会自动设置 runtime library path。手动进入容器排查时可临时设置：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Device extension 构建失败

先保留 Host API 阶段结果：

```bash
SLURM_PARTITION=debug GROUPS=stage0,stage1 BUILD_DEVICE_EXT=0 bash scripts/h20_benchmark.sh
```

Device/Dx 缺失会作为 unsupported 记录，不能把这些行当成性能结论。

legacy Device API extension 构建脚本默认清理旧 build dir 后重建，避免不同 CUDA runtime
产物串用。需要调试增量构建时才使用：

```bash
python native/build_curand_device_extension.py --verbose --no-clean
```
