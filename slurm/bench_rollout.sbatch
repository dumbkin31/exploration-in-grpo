#!/usr/bin/env bash
#SBATCH -J mc-bench
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=3G
#SBATCH --time=02:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH -o slurm-%x-%j.out
# tokens/s and peak memory of vLLM (fp16, TRITON_ATTN) on one 2080 Ti.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh
export MC_JOB_NAME=bench MC_PREFLIGHT_ARGS=--no-staged
mc_job_init
mc_stage_in
mc_run_with_traps python scripts/bench_rollout.py --model "${MC_STAGED_MODEL_DIR}" --data-dir "${MC_STAGED_DATA_DIR}" \
  --concurrency 1 4 8 16 32 --max-tokens "${MC_BENCH_MAX_TOKENS:-1024}" --out "${MC_RUN_DIR}/bench.json"
