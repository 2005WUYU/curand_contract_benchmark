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

SLURM_PARTITION="${SLURM_PARTITION:-debug}"
NUM_GPUS="${NUM_GPUS:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_MB="${MEM_PER_GPU_MB:-32768}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
JOB_NAME="${JOB_NAME:-curand-contract}"

NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
NUM_MEM_MB=$((NUM_GPUS * MEM_PER_GPU_MB))

echo "[h20] partition=${SLURM_PARTITION} gpus=${NUM_GPUS} cpus=${NUM_CPUS} mem=${NUM_MEM_MB}M time=${TIME_LIMIT}"
echo "[h20] image=${IMAGE}"
echo "[h20] repo=${REPO_ROOT}"

srun -p "${SLURM_PARTITION}" \
  --job-name="${JOB_NAME}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="gpu:${NUM_GPUS}" \
  --cpus-per-task="${NUM_CPUS}" \
  --mem="${NUM_MEM_MB}M" \
  --time="${TIME_LIMIT}" \
  --export=ALL,IMAGE,IMAGE_TAR,INNER_CMD,REPO_ROOT \
  bash -lc '
    set -euo pipefail
    echo "[h20] node=$(hostname) step_gpus=${SLURM_STEP_GPUS:-unset}"
    nvidia-smi

    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      echo "[h20] docker image ${IMAGE} not found; loading ${IMAGE_TAR}"
      docker load -i "${IMAGE_TAR}"
    fi

    docker run --rm \
      --gpus all \
      --shm-size=16g \
      -e CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-0}" \
      -v "${REPO_ROOT}":/workspace \
      -w /workspace \
      "${IMAGE}" \
      bash -lc "${INNER_CMD}"
  '
