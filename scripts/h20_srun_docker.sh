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
export CURAND_CONTRACT_CURANDDX_BUILD_DIR="${CURAND_CONTRACT_CURANDDX_BUILD_DIR:-/tmp/curand_contract_curanddx_ext_${USER:-user}}"
export CONTAINER_HOME="${CONTAINER_HOME:-/tmp/curand_contract_home_${USER:-user}}"
export CONTAINER_XDG_CACHE_HOME="${CONTAINER_XDG_CACHE_HOME:-/tmp/curand_contract_cache_${USER:-user}}"
export CONTAINER_TRITON_CACHE_DIR="${CONTAINER_TRITON_CACHE_DIR:-${CONTAINER_XDG_CACHE_HOME}/triton}"
export CONTAINER_TORCH_EXTENSIONS_DIR="${CONTAINER_TORCH_EXTENSIONS_DIR:-${CONTAINER_XDG_CACHE_HOME}/torch_extensions}"

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
echo "[h20] curanddx_build_dir=${CURAND_CONTRACT_CURANDDX_BUILD_DIR}"
echo "[h20] inner_cmd=${INNER_CMD}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[h20] DRY_RUN=1; not submitting srun"
  exit 0
fi

SLURM_EXPORTS="ALL,IMAGE,IMAGE_TAR,INNER_CMD,REPO_ROOT,CURAND_CONTRACT_GIT_SHA"
SLURM_EXPORTS="${SLURM_EXPORTS},CURAND_CONTRACT_DEVICE_BUILD_DIR,CURAND_CONTRACT_CURANDDX_BUILD_DIR"
SLURM_EXPORTS="${SLURM_EXPORTS},CONTAINER_HOME,CONTAINER_XDG_CACHE_HOME"
SLURM_EXPORTS="${SLURM_EXPORTS},CONTAINER_TRITON_CACHE_DIR,CONTAINER_TORCH_EXTENSIONS_DIR"
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
  --export="${SLURM_EXPORTS}" \
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

    prepare_result_target() {
      local candidate="$1"
      local probe="${candidate}/.write_probe_${SLURM_JOB_ID:-$$}"
      mkdir -p "${candidate}" 2>/dev/null || return 1
      touch "${probe}" 2>/dev/null || return 1
      rm -f "${probe}" 2>/dev/null || true
      return 0
    }

    HOST_RESULTS_SPOOL="$(mktemp -d "${TMPDIR:-/tmp}/curand_contract_results_${USER:-user}_XXXXXX")"
    HOST_RESULTS_TARGET="${H20_RESULTS_TARGET:-${REPO_ROOT}/results}"
    if ! prepare_result_target "${HOST_RESULTS_TARGET}"; then
      REQUESTED_RESULTS_TARGET="${HOST_RESULTS_TARGET}"
      HOST_RESULTS_TARGET="${REPO_ROOT}/h20_results"
      echo "[h20] result_target_unwritable=${REQUESTED_RESULTS_TARGET}; falling back to ${HOST_RESULTS_TARGET}" >&2
      if ! prepare_result_target "${HOST_RESULTS_TARGET}"; then
        REQUESTED_RESULTS_TARGET="${HOST_RESULTS_TARGET}"
        HOST_RESULTS_TARGET="${HOME:-/tmp}/curand_contract_benchmark_results"
        echo "[h20] result_target_unwritable=${REQUESTED_RESULTS_TARGET}; falling back to ${HOST_RESULTS_TARGET}" >&2
        if ! prepare_result_target "${HOST_RESULTS_TARGET}"; then
          echo "[h20] no writable result target found; set H20_RESULTS_TARGET to a writable directory" >&2
          exit 126
        fi
      fi
    fi
    echo "[h20] result_spool=${HOST_RESULTS_SPOOL}"
    echo "[h20] result_target=${HOST_RESULTS_TARGET}"

    set +e
    docker run --rm \
      --gpus all \
      --shm-size=16g \
      --user "$(id -u):$(id -g)" \
      -e CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-0}" \
      -e CURAND_CONTRACT_GIT_SHA="${CURAND_CONTRACT_GIT_SHA:-}" \
      -e CURAND_CONTRACT_DEVICE_BUILD_DIR="${CURAND_CONTRACT_DEVICE_BUILD_DIR}" \
      -e CURAND_CONTRACT_CURANDDX_BUILD_DIR="${CURAND_CONTRACT_CURANDDX_BUILD_DIR}" \
      -e HOME="${CONTAINER_HOME}" \
      -e XDG_CACHE_HOME="${CONTAINER_XDG_CACHE_HOME}" \
      -e TRITON_CACHE_DIR="${CONTAINER_TRITON_CACHE_DIR}" \
      -e TORCH_EXTENSIONS_DIR="${CONTAINER_TORCH_EXTENSIONS_DIR}" \
      -e INNER_CMD="${INNER_CMD}" \
      -e MATHDX_ROOT="${MATHDX_ROOT:-}" \
      -e CPATH="${CPATH:-}" \
      -e CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}" \
      -e CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
      -v "${REPO_ROOT}":/workspace \
      -v "${HOST_RESULTS_SPOOL}":/workspace/results \
      -w /workspace \
      "${IMAGE}" \
      bash -lc '"'"'
        set -euo pipefail
        build_ld_library_path() {
          local dirs=()
          local dir
          for dir in \
            /usr/local/cuda/lib64 \
            /usr/local/cuda/lib \
            /usr/local/cuda-*/lib64 \
            /usr/local/cuda-*/lib \
            /opt/conda/lib \
            /opt/conda/lib/python*/site-packages/nvidia/*/lib; do
            if [ -d "${dir}" ]; then
              dirs+=("${dir}")
            fi
          done
          if [ "${#dirs[@]}" -eq 0 ]; then
            return 0
          fi
          local IFS=:
          printf "%s" "${dirs[*]}"
        }
        LD_LIBRARY_PREFIX="$(build_ld_library_path)"
        if [ -n "${LD_LIBRARY_PREFIX}" ]; then
          export LD_LIBRARY_PATH="${LD_LIBRARY_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
        mkdir -p \
          "${HOME}" \
          "${XDG_CACHE_HOME}" \
          "${TRITON_CACHE_DIR}" \
          "${TORCH_EXTENSIONS_DIR}" \
          "${CURAND_CONTRACT_DEVICE_BUILD_DIR}" \
          "${CURAND_CONTRACT_CURANDDX_BUILD_DIR}"
        test -w "${TRITON_CACHE_DIR}"
        echo "[h20] triton_cache=${TRITON_CACHE_DIR}"
        echo "[h20] ld_library_path=${LD_LIBRARY_PATH:-}"
        bash -lc "${INNER_CMD}"
      '"'"'
    docker_rc=$?
    set -e

    if cp -R "${HOST_RESULTS_SPOOL}/." "${HOST_RESULTS_TARGET}/"; then
      echo "[h20] copied results from ${HOST_RESULTS_SPOOL} to ${HOST_RESULTS_TARGET}"
    else
      copy_rc=$?
      echo "[h20] failed to copy results from ${HOST_RESULTS_SPOOL} to ${HOST_RESULTS_TARGET}" >&2
      exit "${copy_rc}"
    fi
    exit "${docker_rc}"
  '
