#!/usr/bin/env bash
# Evaluate on MATH-500, AIME24, AIME25, AMC23 and GPQA-diamond, 16 samples per problem (decision 013).
#
#   bash jarvis/eval.sh base [gpu]                        # the untrained Qwen3-1.7B (reference row)
#   bash jarvis/eval.sh math_grpo-h100-s1 [gpu] [step]    # a run's checkpoint (default: its newest)
#
# A run's LoRA-only checkpoint is merged into the base model in bf16 first (scripts/merge_ckpt.sh).
# Results: <runs>/<run>/eval/global_step_<N>/{summary.json,<benchmark>/results.json}; the base model's
# under <runs>/base-qwen3-1.7b/eval/. scripts/compare_runs.py turns them into the report table.
set -euo pipefail
TARGET="${1:?usage: jarvis/eval.sh base|<run name> [gpu] [step]}"
GPU="${2:-0}"
STEP="${3:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=jarvis
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
export CUDA_VISIBLE_DEVICES="${GPU}" PATH="${MC_VENV_DIR}/bin:${PATH}"
if [ "${TARGET}" = "base" ]; then
  CKPT="${MC_MODEL_DIR}"
  OUT="${MC_RUNS_DIR}/base-qwen3-1.7b/eval/base"
else
  CKDIR="${MC_RUNS_DIR}/${TARGET}/checkpoints"
  [ -d "${CKDIR}" ] || { echo "no checkpoints under ${CKDIR}" >&2; exit 1; }
  STEP="${STEP:-$(cat "${CKDIR}/latest_checkpointed_iteration.txt")}"
  CKPT="${CKDIR}/hf/global_step_${STEP}"
  if [ ! -f "${CKPT}/MERGED_LORA.json" ]; then
    MC_MERGE_DTYPE=bfloat16 bash scripts/merge_ckpt.sh "${CKDIR}" "${STEP}"
  fi
  OUT="${MC_RUNS_DIR}/${TARGET}/eval/global_step_${STEP}"
fi
echo "evaluating ${CKPT} -> ${OUT}"
python eval/run_eval.py --config configs/eval/h100.yaml --ckpt "${CKPT}" --out "${OUT}" --seed 0
