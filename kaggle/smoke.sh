#!/usr/bin/env bash
# Short end-to-end check on a Kaggle T4 (decision 014): the smoke config (2 steps, 4 prompts x 4
# rollouts, 512-token responses) with the T4 layout, then the engine-level GPU tests. ~20 minutes.
#   bash kaggle/smoke.sh [gpu]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=kaggle
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}" MC_SEED=42 RAY_TMPDIR="/tmp/ray-g${GPU}"
export MC_RUN_NAME="smoke-${MC_RUN_TAG}-s42-$$"
export MC_JOB_ID="smoke-$$" MC_JOB_STDOUT="${MC_RUNS_DIR}/${MC_RUN_NAME}.out"
mkdir -p "${MC_RUNS_DIR}" "${RAY_TMPDIR}" "${MC_OUTPUT_DIR}"
echo "smoke run ${MC_RUN_NAME}; log ${MC_JOB_STDOUT}"
rc=0
bash slurm/run_train.sh smoke > "${MC_JOB_STDOUT}" 2>&1 || rc=$?
cp -f "${MC_JOB_STDOUT}" "${MC_OUTPUT_DIR}/" 2>/dev/null || true
cp -f "${MC_RUNS_DIR}/${MC_RUN_NAME}/memory_profile.md" "${MC_OUTPUT_DIR}/${MC_RUN_NAME}.memory_profile.md" 2>/dev/null || true
if [ "${rc}" != 0 ]; then echo "smoke training FAILED (rc=${rc}); tail:"; tail -n 40 "${MC_JOB_STDOUT}"; exit 1; fi
grep -E "step:[0-9]+ -" "${MC_JOB_STDOUT}" | tail -2 | cut -c1-200
echo "training OK; GPU tests:"
MC_SMOKE_RUN_DIR="${MC_RUNS_DIR}/${MC_RUN_NAME}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${MC_VENV_DIR}/bin/python" -m pytest -q tests/gpu -p no:cacheprovider
