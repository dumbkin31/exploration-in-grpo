#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${1:?Usage: $0 <hf-checkpoint> [output-dir]}"
OUT_DIR="${2:-${REPO_ROOT}/runs/saturation/$(basename "${CKPT}")}"
DATA_DIR="${DATA_DIR:-${MC_DATA_DIR:?Set DATA_DIR or MC_DATA_DIR first}}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/default.yaml}"
N_SAMPLES="${N_SAMPLES:-16}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"

DATASETS=(
  math_train
  dapo_train
  # olympiadbench
)

for dataset in "${DATASETS[@]}"; do
  parquet="${DATA_DIR}/${dataset}.parquet"
  if [[ ! -f "${parquet}" ]]; then
    echo "ERROR: missing ${parquet}" >&2
    echo "Prepare the dataset using the verl row schema before running evaluation." >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"

eval_args=(
  --ckpt "${CKPT}"
  --config "${EVAL_CONFIG}"
  --benchmarks "math_train,dapo_train"
  --data-dir "${DATA_DIR}"
  --n-samples "${N_SAMPLES}"
  --seed "${SEED}"
  --out "${OUT_DIR}/eval"
)

if [[ -n "${LIMIT}" ]]; then
  eval_args+=(--limit "${LIMIT}")
fi

echo "Running evaluation..."
"${PYTHON}" eval/run_eval.py "${eval_args[@]}"

echo "Calculating saturation distributions..."
"${PYTHON}" saturation/analyze_saturation.py \
  --eval-dir "${OUT_DIR}/eval" \
  --output-dir "${OUT_DIR}/analysis"

echo "Finished."
echo "Results: ${OUT_DIR}/analysis"