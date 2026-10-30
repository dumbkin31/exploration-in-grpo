#!/usr/bin/env bash
#SBATCH -J mc-saturation
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# Check rollout saturation on MATH training set using Qwen3-0.6B model.
# Submit with: sbatch $(mc_sbatch_args) slurm/saturation.sbatch
# Or: make sbatch-saturation [CKPT=...] [LIMIT=...]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export MC_MODEL_NAME="${MC_MODEL_NAME:-Qwen/Qwen3-0.6B}"
source slurm/common.sh
export MC_PREFLIGHT_ARGS=--no-staged

mc_job_init "saturation-$(basename "${MC_CKPT:-${MC_MODEL_NAME}}")-s${MC_SEED:-0}-${SLURM_JOB_ID}"
mc_stage_in

CKPT="${MC_CKPT:-${MC_STAGED_MODEL_DIR:-${MC_MODEL_DIR}}}"
OUT_DIR="${MC_RUN_DIR}/saturation"
mkdir -p "${OUT_DIR}"

trap 'mc_log "caught signal: flushing"; pkill -TERM -P $$ || true; mc_stage_out; exit 143' USR1 TERM
trap 'mc_stage_out' EXIT

LIMIT="${LIMIT:-${MC_LIMIT:-1600}}"

mc_log "Starting saturation analysis on ${CKPT} with data in ${MC_STAGED_DATA_DIR} (limit: ${LIMIT:-all})"
bash saturation/run_saturation.sh "${CKPT}" "${OUT_DIR}" \
  --datasets "${MC_DATASETS:-math_train}" \
  --gpus "${CUDA_VISIBLE_DEVICES:-0}" \
  --n-samples "${MC_N_SAMPLES:-16}" \
  --seed "${MC_SEED:-0}" \
  ${LIMIT:+--limit "${LIMIT}"} \
  ${MC_SATURATION_ARGS:-}

mc_log "Saturation analysis complete; results in ${OUT_DIR}/analysis"

