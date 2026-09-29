#!/usr/bin/env bash
# Short end-to-end check on the H100 before spending the budget (decision 013): the smoke config
# (2 steps, 4 prompts x 4 rollouts, 512-token responses) with the H100 layout, then the engine-level
# GPU tests. About 15 minutes. Run name smoke-h100-s42-<pid>; it also appears in W&B.
#   bash jarvis/smoke.sh [gpu]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=jarvis
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}" MC_SEED=42 RAY_TMPDIR="/tmp/ray-g${GPU}"
export MC_RUN_NAME="smoke-${MC_RUN_TAG}-s42-$$"
export MC_JOB_ID="smoke-$$" MC_JOB_STDOUT="${MC_RUNS_DIR}/${MC_RUN_NAME}.out"
mkdir -p "${MC_RUNS_DIR}" "${RAY_TMPDIR}"
echo "smoke run ${MC_RUN_NAME}; log ${MC_JOB_STDOUT}"
bash slurm/run_train.sh smoke > "${MC_JOB_STDOUT}" 2>&1 || { echo "smoke training FAILED; tail:"; tail -n 40 "${MC_JOB_STDOUT}"; exit 1; }
grep -E "step:[0-9]+ -" "${MC_JOB_STDOUT}" | tail -2 | cut -c1-200
echo "training OK; GPU tests:"
MC_SMOKE_RUN_DIR="${MC_RUNS_DIR}/${MC_RUN_NAME}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${MC_VENV_DIR}/bin/python" -m pytest -q tests/gpu -p no:cacheprovider
