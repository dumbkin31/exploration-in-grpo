#!/usr/bin/env bash
#SBATCH -J mc-bench
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH -o slurm-%x-%j.out
# Rollout throughput + peak memory: the training layout (one TP=4 engine over 4 GPUs) and, for
# comparison, one TP=1 engine on a single GPU (x4 data-parallel replicas would scale that).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh
export MC_PREFLIGHT_ARGS=--no-staged
mc_job_init "bench-${SLURM_JOB_ID}"
mc_stage_in
trap 'mc_stage_out' EXIT
MAXTOK="${MC_BENCH_MAX_TOKENS:-1024}"
mc_log "== TP=4 (training layout) =="
python scripts/bench_rollout.py --model "${MC_STAGED_MODEL_DIR}" --data-dir "${MC_STAGED_DATA_DIR}" --tp 4 \
  --concurrency 1 8 32 64 128 --max-tokens "${MAXTOK}" --gpu-memory-utilization 0.40 --out "${MC_RUN_DIR}/bench_tp4.json" \
  2>&1 | tee "${MC_JOB_DIR}/bench_tp4.log"
mc_log "== TP=1 on one GPU (for the DP=4 alternative) =="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" python scripts/bench_rollout.py --model "${MC_STAGED_MODEL_DIR}" \
  --data-dir "${MC_STAGED_DATA_DIR}" --tp 1 --concurrency 1 4 8 16 32 --max-tokens "${MAXTOK}" \
  --gpu-memory-utilization 0.70 --out "${MC_RUN_DIR}/bench_tp1.json" 2>&1 | tee "${MC_JOB_DIR}/bench_tp1.log"
grep -hoE "Using [A-Z_]+ attention backend out of potential backends: \[[^]]*\]" "${MC_JOB_DIR}"/bench_tp*.log | sort -u \
  | tee "${MC_RUN_DIR}/attention_backend.txt" || true
