#!/usr/bin/env bash
#SBATCH -J mc-eval
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem-per-cpu=3G
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# Evaluate a HF checkpoint dir on the benchmarks, one vLLM engine per GPU, benchmarks split
# round-robin across GPUs. Submit with: make sbatch-eval CKPT=... BENCH=math500,aime24 N_SAMPLES=16
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh
export MC_JOB_NAME=eval MC_PREFLIGHT_ARGS=--no-staged
mc_job_init
mc_stage_in
CKPT="${MC_CKPT:?set MC_CKPT (HF model dir; use scripts/merge_ckpt.sh for verl checkpoints)}"
BENCH="${MC_BENCH:-math500,aime24,aime25,amc23,gpqa_diamond}"
N_SAMPLES="${MC_N_SAMPLES:-16}"
EVAL_CFG="${MC_EVAL_CONFIG:-configs/eval/default.yaml}"
IFS=',' read -r -a benches <<< "${BENCH}"
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
ngpu=${#gpus[@]}
# Stage out on the SLURM warning signal / termination too, then stop the engines.
trap 'mc_log "caught signal: staging out"; mc_stage_out; pkill -TERM -P $$ || true; exit 143' USR1 TERM
trap 'mc_stage_out' EXIT
pids=()
for i in "${!benches[@]}"; do
  gpu="${gpus[$((i % ngpu))]}"
  b="${benches[$i]}"
  mc_log "benchmark ${b} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python eval/run_eval.py --ckpt "${CKPT}" --benchmarks "${b}" \
      --n-samples "${N_SAMPLES}" --config "${EVAL_CFG}" --data-dir "${MC_STAGED_DATA_DIR}" \
      --out "${MC_RUN_DIR}/eval" > "${MC_RUN_DIR}/eval-${b}.log" 2>&1 &
  pids+=($!)
  # at most one engine per GPU at a time
  if [ $(( (i + 1) % ngpu )) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
python - "${MC_RUN_DIR}/eval" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]); rows = []
for p in sorted(root.glob("*/results.json")):
    r = json.loads(p.read_text()); m = r["metrics"]
    rows.append((r["benchmark"], r["n_problems"], m.get("pass@1"), m.get("pass@16", m.get("pass@4")), m.get("maj@16", m.get("maj@4"))))
print("| benchmark | n | pass@1 | pass@k | maj@k |\n|---|---|---|---|---|")
for b, n, p1, pk, mk in rows:
    f = lambda x: "-" if x is None else f"{100*x:.1f}"
    print(f"| {b} | {n} | {f(p1)} | {f(pk)} | {f(mk)} |")
PY
