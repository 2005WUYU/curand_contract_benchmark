#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SLURM_PARTITION="${SLURM_PARTITION:-long}"
export NUM_GPUS="${NUM_GPUS:-1}"
export CPUS_PER_GPU="${CPUS_PER_GPU:-24}"
export MEM_PER_GPU_MB="${MEM_PER_GPU_MB:-242144}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export JOB_NAME="${JOB_NAME:-curand-h20-benchmark}"

PROFILE="${PROFILE:-h20}"
GROUPS_FROM_ENV="$(printenv GROUPS || true)"
BENCHMARK_GROUPS="${BENCHMARK_GROUPS:-${GROUPS_FROM_ENV:-all}}"
if [[ "${BENCHMARK_GROUPS}" =~ ^[0-9[:space:]]+$ ]]; then
  echo "[h20] ignoring shell GROUPS=${BENCHMARK_GROUPS}; use BENCHMARK_GROUPS or inline GROUPS=all for benchmark task groups" >&2
  BENCHMARK_GROUPS="all"
fi
BUILD_DEVICE_EXT="${BUILD_DEVICE_EXT:-0}"
ALLOW_DEVICE_EXT_FAILURE="${ALLOW_DEVICE_EXT_FAILURE:-1}"

if [ "${NUM_GPUS}" -gt 1 ]; then
  CMD="python scripts/h20_parallel_benchmark.py --profile ${PROFILE} --groups ${BENCHMARK_GROUPS} --num-gpus ${NUM_GPUS}"
else
  CMD="python run_benchmark.py --profile ${PROFILE} --groups ${BENCHMARK_GROUPS}"
fi

if [ "${BUILD_DEVICE_EXT}" = "1" ]; then
  if [ "${ALLOW_DEVICE_EXT_FAILURE}" = "1" ]; then
    CMD="(python native/build_curand_device_extension.py --verbose || true) && ${CMD}"
  else
    CMD="python native/build_curand_device_extension.py --verbose && ${CMD}"
  fi
fi

exec "${SCRIPT_DIR}/h20_srun_docker.sh" "${CMD}"
