#!/usr/bin/env bash
# Evaluate on Kaggle (decision 014): MATH-500, AIME24, AIME25, AMC23, GPQA-diamond, 16 samples per problem,
# fp16 on a T4 (configs/eval/default.yaml, the settings validated for sm_75). ~3 h per model on one T4.
#
#   bash kaggle/eval.sh base [gpu]                        # the untrained Qwen3-1.7B
#   bash kaggle/eval.sh <run name> [gpu] [step]           # e.g. math_grpo-t4-s1, or math_grpo-h100-s1
#
# A run's checkpoint is pulled from the Hub (MC_HUB_REPO) when it is not on disk, merged in fp16, and the
# results are pushed back to the Hub and copied to /kaggle/working/outputs. Two T4s: run two evals at
# once, one per GPU. Evaluate every model you compare on the SAME platform (fp16 T4 vs bf16 H100 differ).
set -euo pipefail
TARGET="${1:?usage: kaggle/eval.sh base|<run name> [gpu] [step]}"
GPU="${2:-0}"
STEP="${3:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=kaggle
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
export CUDA_VISIBLE_DEVICES="${GPU}" PATH="${MC_VENV_DIR}/bin:${PATH}"
hub() { if [ -n "${MC_HUB_REPO:-}" ]; then python scripts/hub_sync.py "$@"; else return 0; fi; }
if [ "${TARGET}" = "base" ]; then
  RUN="base-qwen3-1.7b-${MC_RUN_TAG}"
  CKPT="${MC_MODEL_DIR}"
  OUT="${MC_RUNS_DIR}/${RUN}/eval/base"
else
  RUN="${TARGET}"
  CKDIR="${MC_RUNS_DIR}/${RUN}/checkpoints"
  if [ -z "${STEP}" ] || [ ! -d "${CKDIR}/global_step_${STEP}" ]; then
    hub pull --run "${RUN}" ${STEP:+--step "${STEP}"}
  fi
  [ -d "${CKDIR}" ] || { echo "no checkpoints for ${RUN} (locally or on ${MC_HUB_REPO:-the Hub})" >&2; exit 1; }
  STEP="${STEP:-$(cat "${CKDIR}/latest_checkpointed_iteration.txt")}"
  CKPT="${CKDIR}/hf/global_step_${STEP}"
  [ -f "${CKPT}/MERGED_LORA.json" ] || MC_MERGE_DTYPE=float16 bash scripts/merge_ckpt.sh "${CKDIR}" "${STEP}"
  OUT="${MC_RUNS_DIR}/${RUN}/eval/global_step_${STEP}"
fi
echo "evaluating ${CKPT} -> ${OUT}"
python eval/run_eval.py --config configs/eval/default.yaml --ckpt "${CKPT}" --out "${OUT}" --seed 0
mkdir -p "${MC_RUNS_DIR}/${RUN}"
hub push --run "${RUN}" || echo "WARN: push of the eval results failed"
mkdir -p "${MC_OUTPUT_DIR}/${RUN}/eval" && cp -r "${OUT}" "${MC_OUTPUT_DIR}/${RUN}/eval/" 2>/dev/null || true
