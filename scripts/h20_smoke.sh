#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SLURM_PARTITION="${SLURM_PARTITION:-debug}"
export NUM_GPUS="${NUM_GPUS:-1}"
export CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
export MEM_PER_GPU_MB="${MEM_PER_GPU_MB:-32768}"
export TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
export JOB_NAME="${JOB_NAME:-curand-smoke}"

exec "${SCRIPT_DIR}/h20_srun_docker.sh" \
  "python run_benchmark.py --profile local_smoke"
