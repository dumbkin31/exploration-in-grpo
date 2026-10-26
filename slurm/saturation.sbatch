#!/usr/bin/env bash
#SBATCH -J mc-saturation
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# Check rollout saturation on MATH and DAPO-17k training sets using the Qwen model.
# Submit with: sbatch $(mc_sbatch_args) slurm/saturation.sbatch
# Or: make sbatch-saturation [CKPT=...] [LIMIT=...]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh
export MC_PREFLIGHT_ARGS=--no-staged

CKPT="${MC_CKPT:-${MC_MODEL_DIR}}"
mc_job_init "saturation-$(basename "${CKPT}")-s${MC_SEED:-0}-${SLURM_JOB_ID}"
mc_stage_in

OUT_DIR="${MC_RUN_DIR}/saturation"
mkdir -p "${OUT_DIR}"

trap 'mc_log "caught signal: flushing"; pkill -TERM -P $$ || true; mc_stage_out; exit 143' USR1 TERM
trap 'mc_stage_out' EXIT

mc_log "Starting saturation analysis on ${CKPT} with data in ${MC_STAGED_DATA_DIR}"
bash saturation/run_saturation.sh "${CKPT}" "${OUT_DIR}" \
  --datasets "${MC_DATASETS:-math_train,dapo_train}" \
  --gpus "${CUDA_VISIBLE_DEVICES:-0}" \
  --n-samples "${MC_N_SAMPLES:-16}" \
  --seed "${MC_SEED:-0}" \
  ${MC_SATURATION_ARGS:-}

mc_log "Saturation analysis complete; results in ${OUT_DIR}/analysis"

