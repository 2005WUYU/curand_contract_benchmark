#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SLURM_PARTITION="${SLURM_PARTITION:-long}"
export NUM_GPUS="${NUM_GPUS:-1}"
export CPUS_PER_GPU="${CPUS_PER_GPU:-24}"
export MEM_PER_GPU_MB="${MEM_PER_GPU_MB:-242144}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export JOB_NAME="${JOB_NAME:-curand-h20-hostapi-only}"

PROFILE="${PROFILE:-h20}"
CMD="python scripts/verify_flagrand_vendor.py && python scripts/hostapi_only_benchmark.py --profile ${PROFILE} --num-gpus ${NUM_GPUS}"

if [ -n "${HOSTAPI_GENERATORS:-}" ]; then
  CMD="${CMD} --generators ${HOSTAPI_GENERATORS}"
fi
if [ -n "${HOSTAPI_DISTRIBUTIONS:-}" ]; then
  CMD="${CMD} --distributions ${HOSTAPI_DISTRIBUTIONS}"
fi
if [ -n "${HOSTAPI_SIZES:-}" ]; then
  CMD="${CMD} --sizes ${HOSTAPI_SIZES}"
fi
if [ -n "${HOSTAPI_POISSON_LAMBDAS:-}" ]; then
  CMD="${CMD} --poisson-lambdas ${HOSTAPI_POISSON_LAMBDAS}"
fi
if [ -n "${HOSTAPI_WARMUP:-}" ]; then
  CMD="${CMD} --warmup ${HOSTAPI_WARMUP}"
fi
if [ -n "${HOSTAPI_REPEATS:-}" ]; then
  CMD="${CMD} --repeats ${HOSTAPI_REPEATS}"
fi
if [ -n "${HOSTAPI_SEED:-}" ]; then
  CMD="${CMD} --seed ${HOSTAPI_SEED}"
fi
if [ -n "${HOSTAPI_OFFSET:-}" ]; then
  CMD="${CMD} --offset ${HOSTAPI_OFFSET}"
fi
if [ -n "${HOSTAPI_ORDERING:-}" ]; then
  CMD="${CMD} --ordering ${HOSTAPI_ORDERING}"
fi
if [ -n "${HOSTAPI_QRNG_DIMENSIONS:-}" ]; then
  CMD="${CMD} --qrng-dimensions ${HOSTAPI_QRNG_DIMENSIONS}"
fi
if [ -n "${HOSTAPI_MAX_CASES:-}" ]; then
  CMD="${CMD} --max-cases ${HOSTAPI_MAX_CASES}"
fi
if [ -n "${HOSTAPI_RESULTS_DIR:-}" ]; then
  CMD="${CMD} --results-dir ${HOSTAPI_RESULTS_DIR}"
fi

exec "${SCRIPT_DIR}/h20_srun_docker.sh" "${CMD}"
