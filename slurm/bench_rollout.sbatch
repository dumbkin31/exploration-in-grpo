#!/usr/bin/env bash
#SBATCH -J mc-bench
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=03:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH -o slurm-%x-%j.out
# Rollout throughput + peak memory for the layout that will train (decision 012): one TP=1 engine on one
# GPU with the training-time engine shape (max_model_len 6144, chunked prefill 4096), a concurrency
# sweep at 1024 tokens, a long-response probe at 4096 tokens (preemption behaviour), and a start-up
# probe at util 0.85. Under MC_LAYOUT=nlp_4gpu the TP=4 sweep of decision 002 runs first.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh
export MC_PREFLIGHT_ARGS=--no-staged
mc_job_init "bench-${SLURM_JOB_ID}"
mc_stage_in
trap 'mc_stage_out' EXIT
MAXTOK="${MC_BENCH_MAX_TOKENS:-1024}"
UTIL="${MC_BENCH_UTIL:-0.80}"
GPU0="${CUDA_VISIBLE_DEVICES%%,*}"
COMMON=(--model "${MC_STAGED_MODEL_DIR}" --data-dir "${MC_STAGED_DATA_DIR}" --max-model-len 6144
        --max-num-seqs "${MC_BENCH_MAX_NUM_SEQS:-64}" --max-num-batched-tokens 4096)
if [ "${MC_LAYOUT}" = "nlp_4gpu" ]; then
  mc_log "== TP=4 (nlp_4gpu layout) =="
  python scripts/bench_rollout.py "${COMMON[@]}" --tp 4 --concurrency 1 8 32 64 128 --max-tokens "${MAXTOK}" \
    --gpu-memory-utilization 0.40 --out "${MC_RUN_DIR}/bench_tp4.json" 2>&1 | tee "${MC_JOB_DIR}/bench_tp4.log"
fi
mc_log "== TP=1, util ${UTIL}, ${MAXTOK} tokens: the research_1gpu rollout shape =="
CUDA_VISIBLE_DEVICES="${GPU0}" python scripts/bench_rollout.py "${COMMON[@]}" --tp 1 --concurrency 1 8 16 32 64 \
  --max-tokens "${MAXTOK}" --gpu-memory-utilization "${UTIL}" --out "${MC_RUN_DIR}/bench_tp1.json" 2>&1 | tee "${MC_JOB_DIR}/bench_tp1.log"
mc_log "== TP=1, util ${UTIL}, 4096 tokens: long responses (KV pressure, preemptions) =="
CUDA_VISIBLE_DEVICES="${GPU0}" python scripts/bench_rollout.py "${COMMON[@]}" --tp 1 --concurrency 8 16 32 \
  --max-tokens 4096 --gpu-memory-utilization "${UTIL}" --out "${MC_RUN_DIR}/bench_tp1_long.json" 2>&1 | tee "${MC_JOB_DIR}/bench_tp1_long.log"
mc_log "== TP=1 start-up probe at util 0.85 =="
CUDA_VISIBLE_DEVICES="${GPU0}" python scripts/bench_rollout.py "${COMMON[@]}" --tp 1 --concurrency 1 --max-tokens 32 \
  --gpu-memory-utilization 0.85 --out "${MC_RUN_DIR}/bench_tp1_util085.json" 2>&1 | tee "${MC_JOB_DIR}/bench_tp1_util085.log" \
  || mc_log "WARN: the engine did not start at util 0.85 on this node; keep 0.80"
grep -hoE "Using [A-Z_]+ attention backend out of potential backends: \[[^]]*\]|GPU KV cache size: [^,]+|Maximum concurrency for [^:]+: [^ ]+|[0-9]+ preempt[a-z]*|Sleep mode [a-z ]+" \
  "${MC_JOB_DIR}"/bench_tp*.log | sort | uniq -c | tee "${MC_RUN_DIR}/bench_engine_facts.txt" || true
