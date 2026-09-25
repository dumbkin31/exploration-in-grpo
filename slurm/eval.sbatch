#!/usr/bin/env bash
#SBATCH -J mc-eval
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem-per-cpu=3G
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# Evaluate a HF checkpoint dir on the benchmarks, one vLLM engine per GPU, benchmarks split
# round-robin across GPUs. Submit with: make sbatch-eval CKPT=... BENCH=math500,aime24 N_SAMPLES=16 SEED=0
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh
export MC_PREFLIGHT_ARGS=--no-staged
CKPT="${MC_CKPT:?set MC_CKPT (HF model dir; use scripts/merge_ckpt.sh for verl checkpoints)}"
mc_job_init "eval-$(basename "${CKPT}")-s${MC_SEED:-0}-${SLURM_JOB_ID}"
mc_stage_in
BENCH="${MC_BENCH:-math500,aime24,aime25,amc23,gpqa_diamond}"
N_SAMPLES="${MC_N_SAMPLES:-16}"
EVAL_CFG="${MC_EVAL_CONFIG:-configs/eval/default.yaml}"
IFS=',' read -r -a benches <<< "${BENCH}"
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
ngpu=${#gpus[@]}
trap 'mc_log "caught signal: flushing"; pkill -TERM -P $$ || true; mc_stage_out; exit 143' USR1 TERM
trap 'mc_stage_out' EXIT
pids=()
for i in "${!benches[@]}"; do
  gpu="${gpus[$((i % ngpu))]}"
  b="${benches[$i]}"
  mc_log "benchmark ${b} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python eval/run_eval.py --ckpt "${CKPT}" --benchmarks "${b}" \
      --n-samples "${N_SAMPLES}" --config "${EVAL_CFG}" --data-dir "${MC_STAGED_DATA_DIR}" --seed "${MC_SEED:-0}" \
      --out "${MC_RUN_DIR}/eval" > "${MC_JOB_DIR}/eval-${b}.log" 2>&1 &
  pids+=($!)
  if [ $(( (i + 1) % ngpu )) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
python - "${MC_RUN_DIR}/eval" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]); rows = []
for p in sorted(root.glob("*/results.json")):
    r = json.loads(p.read_text()); m = r["metrics"]
    rows.append((r["benchmark"], r["n_problems"], m.get("pass@1"), m.get("pass@1_ci95"), m.get("pass@16", m.get("pass@4")), m.get("maj@16", m.get("maj@4")), r.get("notes", "")))
print("| benchmark | n | pass@1 [95% CI] | pass@k | maj@k |\n|---|---|---|---|---|")
f = lambda x: "-" if x is None else f"{100*x:.1f}"
for b, n, p1, ci, pk, mk, note in rows:
    cis = f" [{100*ci[0]:.1f}, {100*ci[1]:.1f}]" if ci else ""
    print(f"| {b} | {n} | {f(p1)}{cis} | {f(pk)} | {f(mk)} |")
for b, n, *_rest, note in rows:
    print(f"- {note}")
PY
