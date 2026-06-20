#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-local_smoke}"
GROUPS="${BENCHMARK_GATE_GROUPS:-G0_BASIC_CONTRACT,G1_DISTRIBUTION_ROUGH_CHECK,G2_REPRODUCIBILITY,G3_SEQUENCE_COUNTER_BUDGET}"

python run_benchmark.py --profile "${PROFILE}" --groups "${GROUPS}"
