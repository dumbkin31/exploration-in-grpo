#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_saturation.sh <hf-checkpoint> [output-dir] [options]

Options:
  --start N          first dataset row, inclusive (default: 0)
  --end N            last dataset row, exclusive (default: dataset length)
  --datasets CSV     datasets to run (default: math_train,dapo_train)
  --gpus CSV         visible GPU ids, assigned round-robin (default: 0)
  --chunk-size N     rows per resumable evaluation chunk (default: 100)
  --limit N           number of rows from --start (compatibility shortcut)
  --n-samples N      rollouts per example (default: 16)
  --seed N            evaluation seed (default: 0)
  --no-analyze       skip aggregation; use after parallel chunk workers
  --analyze          aggregate all completed chunks after evaluation (default)
  --analyze-only     aggregate existing chunks without running evaluation
EOF
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${1:-}"
[[ -n "${CKPT}" ]] || usage
shift
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  OUT_DIR="$1"
  shift
else
  OUT_DIR="${REPO_ROOT}/runs/saturation/$(basename "${CKPT}")"
fi

DATA_DIR="${DATA_DIR:-${MC_DATA_DIR:-}}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/default.yaml}"
DATASETS_CSV="${DATASETS:-math_train,dapo_train}"
GPUS_CSV="${GPUS:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
START="${START:-0}"
END="${END:-}"
N_SAMPLES="${N_SAMPLES:-16}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"
ANALYZE=1
ANALYZE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --no-analyze) ANALYZE=0; shift ;;
    --analyze) ANALYZE=1; shift ;;
    --analyze-only) ANALYZE_ONLY=1; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ "${START}" =~ ^[0-9]+$ ]] || { echo "ERROR: --start must be a non-negative integer" >&2; exit 2; }
[[ "${CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --chunk-size must be a positive integer" >&2; exit 2; }

if [[ -n "${LIMIT}" ]]; then
  [[ "${LIMIT}" =~ ^[0-9]+$ ]] || { echo "ERROR: --limit must be a non-negative integer" >&2; exit 2; }
  if [[ -n "${END}" ]]; then
    echo "ERROR: use either --limit or --end, not both" >&2
    exit 2
  fi
  END=$((START + LIMIT))
fi

if [[ -n "${END}" && ! "${END}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --end must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${END}" ]] && (( END < START )); then
  echo "ERROR: --end must be greater than or equal to --start" >&2
  exit 2
fi

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "ERROR: --gpus cannot be empty" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "ERROR: GPU ids must be non-negative integers" >&2; exit 2; }
done

mkdir -p "${OUT_DIR}/chunks" "${OUT_DIR}/eval"

if (( ANALYZE_ONLY )); then
  "${PYTHON}" saturation/analyze_saturation.py \
    --eval-dir "${OUT_DIR}/eval" \
    --output-dir "${OUT_DIR}/analysis" \
    --datasets "${DATASETS[@]}"
  echo "Results: ${OUT_DIR}/analysis"
  exit 0
fi

if [[ -z "${DATA_DIR}" ]]; then
  echo "ERROR: set DATA_DIR or MC_DATA_DIR before evaluation" >&2
  exit 2
fi

for dataset in "${DATASETS[@]}"; do
  parquet="${DATA_DIR}/${dataset}.parquet"
  if [[ ! -f "${parquet}" ]]; then
    echo "ERROR: missing ${parquet}" >&2
    echo "Prepare the dataset using the verl row schema before running evaluation." >&2
    exit 1
  fi
done

run_dataset() {
  local dataset="$1"
  local gpu="$2"
  local dataset_end="${END}"
  local total_rows
  total_rows=$("${PYTHON}" - "${DATA_DIR}/${dataset}.parquet" <<'PY'
import sys

import pyarrow.parquet as parquet

print(parquet.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
  )
  if [[ -z "${dataset_end}" || "${dataset_end}" -gt "${total_rows}" ]]; then
    dataset_end="${total_rows}"
  fi
  if (( START > total_rows )); then
    echo "ERROR: --start=${START} exceeds ${total_rows} rows in ${dataset}" >&2
    return 1
  fi

  local chunk_start="${START}"
  while (( chunk_start < dataset_end )); do
    local chunk_end=$((chunk_start + CHUNK_SIZE))
    (( chunk_end > dataset_end )) && chunk_end="${dataset_end}"
    local range="${chunk_start}-${chunk_end}"
    local dataset_chunk_dir="${OUT_DIR}/chunks/${dataset}-${range}"
    local eval_root="${OUT_DIR}/eval/${dataset}/${range}"
    local result_dir="${eval_root}/${dataset}"
    mkdir -p "${dataset_chunk_dir}" "${eval_root}"

    if [[ -f "${result_dir}/results.json" && -f "${result_dir}/samples.jsonl" ]]; then
      echo "Skipping completed ${dataset} rows [${chunk_start},${chunk_end}) on GPU ${gpu}"
      chunk_start="${chunk_end}"
      continue
    fi

    local dataset_chunk_file="${dataset_chunk_dir}/${dataset}.parquet"
    if [[ ! -f "${dataset_chunk_file}" ]]; then
      "${PYTHON}" - "${DATA_DIR}/${dataset}.parquet" "${dataset_chunk_file}" "${chunk_start}" "${chunk_end}" <<'PY'
import sys

import pandas as pd

source, target, start, end = sys.argv[1:]
frame = pd.read_parquet(source)
start, end = int(start), int(end)
if start > len(frame):
    raise SystemExit(f"start={start} exceeds {len(frame)} rows in {source}")
frame.iloc[start:end].to_parquet(target, index=False)
print(f"wrote {target}: rows [{start}, {min(end, len(frame))}) of {len(frame)}")
PY
    fi

    echo "Running ${dataset} rows [${chunk_start},${chunk_end}) on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" eval/run_eval.py \
      --ckpt "${CKPT}" \
      --config "${EVAL_CONFIG}" \
      --benchmarks "${dataset}" \
      --data-dir "${dataset_chunk_dir}" \
      --n-samples "${N_SAMPLES}" \
      --seed "${SEED}" \
      --out "${eval_root}"
    chunk_start="${chunk_end}"
  done
}

if (( ${#GPUS[@]} >= ${#DATASETS[@]} )); then
  declare -a PIDS=()
  for i in "${!DATASETS[@]}"; do
    run_dataset "${DATASETS[$i]}" "${GPUS[$i]}" &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
else
  for i in "${!DATASETS[@]}"; do
    run_dataset "${DATASETS[$i]}" "${GPUS[$((i % ${#GPUS[@]}))]}"
  done
fi

if (( ANALYZE )); then
  echo "Calculating saturation distributions from completed chunks..."
  "${PYTHON}" saturation/analyze_saturation.py \
    --eval-dir "${OUT_DIR}/eval" \
    --output-dir "${OUT_DIR}/analysis" \
    --datasets "${DATASETS[@]}"
  echo "Results: ${OUT_DIR}/analysis"
else
  echo "Evaluation chunks complete; run again with --analyze after all workers finish."
fi