#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 '<command to run inside container>'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export IMAGE="${IMAGE:-flagtree-nvidia:3.6-v2}"
export IMAGE_TAR="${IMAGE_TAR:-/data/nfs3/flagtree-nvidia-3.6-v2.tar}"
export INNER_CMD="$*"
export REPO_ROOT
export CURAND_CONTRACT_GIT_SHA="${CURAND_CONTRACT_GIT_SHA:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)}"
export CURAND_CONTRACT_DEVICE_BUILD_DIR="${CURAND_CONTRACT_DEVICE_BUILD_DIR:-/tmp/curand_contract_device_ext_${USER:-user}}"

SLURM_PARTITION="${SLURM_PARTITION:-debug}"
SLURM_NODELIST="${SLURM_NODELIST:-${H20_NODELIST:-}}"
NUM_GPUS="${NUM_GPUS:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_MB="${MEM_PER_GPU_MB:-32768}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
JOB_NAME="${JOB_NAME:-curand-contract}"

NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
NUM_MEM_MB=$((NUM_GPUS * MEM_PER_GPU_MB))

echo "[h20] partition=${SLURM_PARTITION} gpus=${NUM_GPUS} cpus=${NUM_CPUS} mem=${NUM_MEM_MB}M time=${TIME_LIMIT}"
if [ -n "${SLURM_NODELIST}" ]; then
  echo "[h20] nodelist=${SLURM_NODELIST}"
fi
echo "[h20] image=${IMAGE}"
echo "[h20] image_tar=${IMAGE_TAR}"
echo "[h20] repo=${REPO_ROOT}"
echo "[h20] git_sha=${CURAND_CONTRACT_GIT_SHA:-unknown}"
echo "[h20] device_build_dir=${CURAND_CONTRACT_DEVICE_BUILD_DIR}"
echo "[h20] inner_cmd=${INNER_CMD}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[h20] DRY_RUN=1; not submitting srun"
  exit 0
fi

SRUN_CMD=(srun -p "${SLURM_PARTITION}")
if [ -n "${SLURM_NODELIST}" ]; then
  SRUN_CMD+=(--nodelist="${SLURM_NODELIST}")
fi

"${SRUN_CMD[@]}" \
  --job-name="${JOB_NAME}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="gpu:${NUM_GPUS}" \
  --cpus-per-task="${NUM_CPUS}" \
  --mem="${NUM_MEM_MB}M" \
  --time="${TIME_LIMIT}" \
  --export=ALL,IMAGE,IMAGE_TAR,INNER_CMD,REPO_ROOT,CURAND_CONTRACT_GIT_SHA \
  bash -lc '
    set -euo pipefail
    echo "[h20] node=$(hostname) step_gpus=${SLURM_STEP_GPUS:-unset}"
    nvidia-smi

    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      echo "[h20] docker image ${IMAGE} not found; loading ${IMAGE_TAR}"
      docker load -i "${IMAGE_TAR}"
    fi
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      echo "[h20] docker image ${IMAGE} is still missing after loading ${IMAGE_TAR}" >&2
      echo "[h20] set IMAGE_TAR to a tar that contains ${IMAGE}, or use H20_NODELIST/SLURM_NODELIST for a node that already has it" >&2
      exit 125
    fi

    docker run --rm \
      --gpus all \
      --shm-size=16g \
      -e CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-0}" \
      -e CURAND_CONTRACT_GIT_SHA="${CURAND_CONTRACT_GIT_SHA:-}" \
      -e CURAND_CONTRACT_DEVICE_BUILD_DIR="${CURAND_CONTRACT_DEVICE_BUILD_DIR}" \
      -e MATHDX_ROOT="${MATHDX_ROOT:-}" \
      -e CPATH="${CPATH:-}" \
      -e CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}" \
      -e CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
      -v "${REPO_ROOT}":/workspace \
      -w /workspace \
      "${IMAGE}" \
      bash -lc "${INNER_CMD}"
  '
