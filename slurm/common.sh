#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared job plumbing for every slurm/*.sbatch and slurm/run_train.sh. Source it, then call:
#
#   mc_job_init [RUN_NAME]   paths, env, preflight (aborts early on FAIL), GPU memory sampler
#   mc_stage_in              model + parquet from $MC_STAGE_ROOT -> node-local /scratch
#   mc_run_with_traps CMD    run CMD; on SIGUSR1/SIGTERM/EXIT copy job logs out and prune checkpoints
#   mc_stage_out             idempotent copy of job logs into the durable run dir + engine-log facts
#   mc_train_main CONFIG [overrides...]   the whole training job (used by run_train.sh)
#
# Layout (docs/decisions/005):
#   MC_RUN_NAME    = <config>-s<seed>            stable across resubmissions (override: MC_RUN_NAME=...)
#   MC_RUN_DIR     = $MC_RUNS_DIR/$MC_RUN_NAME   DURABLE (/share1): checkpoints/, metrics.jsonl,
#                                                cuts_stats/, rollout_dumps/, phases.jsonl,
#                                                gpu_mem.jsonl, wandb/, jobs/<jobid>/
#   MC_SCRATCH_ROOT (node-local)                 staged model/data copies and caches ONLY
# verl's resume_mode=auto reads $MC_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt, so a
# resubmitted job continues where the last periodic checkpoint left off (verl cannot checkpoint
# on SIGTERM; the grace period only flushes logs). WANDB_RUN_ID = run name keeps one W&B run.
# ---------------------------------------------------------------------------
set -euo pipefail

_MC_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$(cd "${_MC_COMMON_DIR}/.." && pwd)}"

mc_log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

mc_job_init() {
  # shellcheck source=/dev/null
  source "${MC_REPO_ROOT}/configs/ada.env.sh"
  export MC_JOB_ID="${SLURM_JOB_ID:-local-$$}"
  export MC_SEED="${MC_SEED:-42}"
  export MC_RUN_NAME="${1:-${MC_RUN_NAME:-${SLURM_JOB_NAME:-job}-s${MC_SEED}}}"
  export MC_RUN_DIR="${MC_RUN_DIR:-${MC_RUNS_DIR}/${MC_RUN_NAME}}"          # durable (/share1)
  export MC_JOB_DIR="${MC_RUN_DIR}/jobs/${MC_JOB_ID}"                        # this submission's logs
  export MC_STAGED_MODEL_DIR="${MC_SCRATCH_ROOT}/stage/models/$(basename "${MC_MODEL_DIR}")"
  export MC_STAGED_DATA_DIR="${MC_SCRATCH_ROOT}/stage/data"
  export MC_ENV_FACTS_JSON="${MC_JOB_DIR}/env_facts.json"
  export MC_JOB_STDOUT="${SLURM_SUBMIT_DIR:-$PWD}/slurm-${SLURM_JOB_NAME:-job}-${MC_JOB_ID}.out"   # matches -o slurm-%x-%j.out
  # W&B: offline, one run id across resubmissions (WANDB_RESUME=allow continues the same curves;
  # steps re-logged after a kill are dropped by W&B; metrics.jsonl is the primary log).
  export WANDB_DIR="${MC_RUN_DIR}/wandb"
  export WANDB_RUN_ID="${MC_RUN_NAME}"
  export WANDB_RESUME="allow"
  export WANDB_NAME="${MC_RUN_NAME}"
  mkdir -p "${MC_RUN_DIR}" "${MC_JOB_DIR}" "${WANDB_DIR}" "${MC_SCRATCH_ROOT}/stage" \
    || { mc_log "ERROR: cannot create ${MC_RUN_DIR} (is ${MC_STAGE_ROOT} mounted and writable here?)"; exit 3; }
  export PATH="${MC_VENV_DIR}/bin:${PATH}"
  export PYTHONUNBUFFERED=1

  mc_log "job ${MC_JOB_ID} run ${MC_RUN_NAME} on $(hostname); run dir ${MC_RUN_DIR}; job dir ${MC_JOB_DIR}"
  mc_log "python: $(command -v python) ; GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ; nodes: ${SLURM_JOB_NUM_NODES:-?}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
  {
    echo "job_id=${MC_JOB_ID}"; echo "run_name=${MC_RUN_NAME}"; echo "seed=${MC_SEED}"; echo "node=$(hostname)"
    echo "git=$(git -C "${MC_REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo ?)"; date
  } > "${MC_JOB_DIR}/job_info.txt"

  # Preflight: pins, sm_75, single node, engine version, storage, internet. Aborts on FAIL.
  python "${MC_REPO_ROOT}/scripts/check_env.py" ${MC_PREFLIGHT_ARGS:---no-staged} --facts-json "${MC_ENV_FACTS_JSON}" \
    | tee "${MC_JOB_DIR}/preflight.log"
  test "${PIPESTATUS[0]}" -eq 0 || { mc_log "preflight FAILED; see ${MC_JOB_DIR}/preflight.log"; exit 4; }
  if python - "$MC_ENV_FACTS_JSON" <<'PY'
import json, sys
try:
    facts = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if facts.get("internet", {}).get("https://huggingface.co") is False else 1)
PY
  then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    mc_log "no internet on this node: HF_HUB_OFFLINE=1"
  fi
  mc_start_gpu_sampler
}

mc_start_gpu_sampler() {
  # Per-GPU memory timeline for scripts/profile_memory.py --report (joined with phases.jsonl).
  python "${MC_REPO_ROOT}/scripts/profile_memory.py" --watch "${MC_RUN_DIR}/gpu_mem.jsonl" --job "${MC_JOB_ID}" \
    > "${MC_JOB_DIR}/gpu_sampler.log" 2>&1 &
  export _MC_SAMPLER_PID=$!
}

mc_stop_gpu_sampler() {
  if [ -n "${_MC_SAMPLER_PID:-}" ] && kill -0 "${_MC_SAMPLER_PID}" 2>/dev/null; then
    kill "${_MC_SAMPLER_PID}" 2>/dev/null || true
    wait "${_MC_SAMPLER_PID}" 2>/dev/null || true
  fi
  _MC_SAMPLER_PID=""
}

mc_stage_in() {
  if [ ! -d "${MC_STAGE_ROOT}" ]; then
    mc_log "ERROR: ${MC_STAGE_ROOT} is not visible on $(hostname). Export MC_STAGE_ROOT to a path this node can read."
    exit 3
  fi
  if [ ! -f "${MC_MODEL_DIR}/config.json" ]; then
    mc_log "ERROR: model not staged at ${MC_MODEL_DIR}; run 'make prefetch' on the login node"; exit 3
  fi
  mkdir -p "${MC_STAGED_MODEL_DIR}" "${MC_STAGED_DATA_DIR}"
  mc_log "stage-in model -> ${MC_STAGED_MODEL_DIR}"
  rsync -a --info=stats1 "${MC_MODEL_DIR}/" "${MC_STAGED_MODEL_DIR}/"
  if [ -d "${MC_DATA_DIR}" ]; then
    mc_log "stage-in data  -> ${MC_STAGED_DATA_DIR}"
    rsync -a --info=stats1 "${MC_DATA_DIR}/" "${MC_STAGED_DATA_DIR}/"
  else
    mc_log "WARN: ${MC_DATA_DIR} missing (run 'make data'); continuing without data"
  fi
}

mc_engine_log_facts() {
  # vLLM logs its choices through Ray to the driver stdout (= this job's .out file). Record them.
  local out="${MC_JOB_STDOUT}"
  [ -f "${out}" ] || return 0
  local attn mrv2
  attn="$(grep -oE "Using [A-Z_]+ attention backend out of potential backends: \[[^]]*\]" "${out}" | head -1 || true)"
  if grep -q "Using V2 Model Runner" "${out}"; then mrv2=true; else mrv2=false; fi
  mc_log "engine facts: attention='${attn:-not found in log}' model_runner_v2=${mrv2}"
  python - "${MC_ENV_FACTS_JSON}" "${attn}" "${mrv2}" <<'PY'
import json, sys, os
path, attn, mrv2 = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
facts = json.load(open(path)) if os.path.exists(path) else {}
facts["vllm_attention_backend_log"] = attn or None
facts["vllm_model_runner_v2"] = mrv2
json.dump(facts, open(path, "w"), indent=2)
PY
  if [ "${mrv2}" = true ]; then
    mc_log "ERROR: vLLM used Model Runner V2; the CUTS logits processor needs V1 (custom LPs force V1 unless VLLM_USE_V2_MODEL_RUNNER=1 is set)"
  fi
}

mc_prune_checkpoints() {
  # verl's max_actor_ckpt_to_keep never prunes checkpoints saved before a restart (docs/decisions/005).
  # Keep the step named in latest_checkpointed_iteration.txt plus the previous one.
  local ckdir="${MC_RUN_DIR}/checkpoints" keep="${MC_KEEP_CHECKPOINTS:-2}"
  [ -d "${ckdir}" ] || return 0
  local latest=""
  [ -f "${ckdir}/latest_checkpointed_iteration.txt" ] && latest="$(cat "${ckdir}/latest_checkpointed_iteration.txt")"
  local dirs
  dirs="$(ls -d "${ckdir}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)"
  local n; n="$(echo "${dirs}" | grep -c . || true)"
  [ "${n}" -gt "${keep}" ] || return 0
  echo "${dirs}" | head -n "$((n - keep))" | while read -r step; do
    [ -n "${step}" ] || continue
    [ "${step}" = "${latest}" ] && continue
    mc_log "pruning old checkpoint global_step_${step}"
    rm -rf "${ckdir}/global_step_${step}"
  done
}

mc_stage_out() {
  # Idempotent. Outputs already live in the durable run dir; this copies THIS job's logs next to
  # them, records the engine facts and prunes stale checkpoints. Safe to call from the trap and EXIT.
  mc_stop_gpu_sampler
  mc_engine_log_facts || true
  if [ -f "${MC_JOB_STDOUT}" ]; then cp -f "${MC_JOB_STDOUT}" "${MC_JOB_DIR}/" 2>/dev/null || true; fi
  mc_prune_checkpoints || true
  mc_log "stage-out done (run dir ${MC_RUN_DIR})"
}

_MC_CHILD_PID=""
_mc_on_signal() {
  local sig="$1"
  mc_log "caught ${sig}: flushing (child pid ${_MC_CHILD_PID:-none}); the run resumes from its last checkpoint"
  if [ -n "${_MC_CHILD_PID}" ] && kill -0 "${_MC_CHILD_PID}" 2>/dev/null; then
    kill -TERM "${_MC_CHILD_PID}" 2>/dev/null || true
    for _ in $(seq 1 90); do kill -0 "${_MC_CHILD_PID}" 2>/dev/null || break; sleep 2; done
    kill -KILL "${_MC_CHILD_PID}" 2>/dev/null || true
  fi
  mc_stage_out
  exit 143
}

_mc_kill_watcher() {
  # Test hook (slurm/test_resume.sbatch): send this shell SIGUSR1 once metrics.jsonl reaches a step,
  # which exercises the real SLURM signal path end to end.
  local target="$1" parent="$2"
  while sleep 10; do
    kill -0 "${parent}" 2>/dev/null || exit 0
    if [ -f "${MC_RUN_DIR}/metrics.jsonl" ] && grep -qE "\"step\": ?${target}[,}]" "${MC_RUN_DIR}/metrics.jsonl"; then
      mc_log "kill watcher: step ${target} logged; sending SIGUSR1 to ${parent}"
      kill -USR1 "${parent}"
      exit 0
    fi
  done
}

mc_run_with_traps() {
  trap '_mc_on_signal SIGUSR1' USR1
  trap '_mc_on_signal SIGTERM' TERM
  trap 'mc_stage_out' EXIT
  if [ -n "${MC_KILL_AFTER_STEP:-}" ]; then
    _mc_kill_watcher "${MC_KILL_AFTER_STEP}" "$$" &
  fi
  mc_log "launch: $*"
  "$@" &
  _MC_CHILD_PID=$!
  local rc=0
  wait "${_MC_CHILD_PID}" || rc=$?
  _MC_CHILD_PID=""
  mc_log "command finished with rc=${rc}"
  return "${rc}"
}

mc_train_main() {
  # Usage: mc_train_main <config-name> [extra hydra overrides...]; needs MC_SEED (default 42).
  local config="${1:?config name}"; shift || true
  export MC_SEED="${MC_SEED:-42}"
  mc_job_init "${MC_RUN_NAME:-${config}-s${MC_SEED}}"
  mc_stage_in
  # Config-specific preflight (compose + invariants + chat template) now that the model is staged.
  python "${MC_REPO_ROOT}/scripts/check_env.py" --no-pins --no-gpu --no-staged --only-config \
    --config "${config}" --model-dir "${MC_STAGED_MODEL_DIR}" | tee -a "${MC_JOB_DIR}/preflight.log"
  test "${PIPESTATUS[0]}" -eq 0 || { mc_log "config preflight FAILED"; exit 4; }
  local extra=()
  [ -n "${MC_SAVE_FREQ:-}" ] && extra+=("save_freq=${MC_SAVE_FREQ}")
  [ -n "${MC_TOTAL_STEPS:-}" ] && extra+=("trainer.total_training_steps=${MC_TOTAL_STEPS}")
  cd "${MC_REPO_ROOT}"
  mc_run_with_traps python -m mixed_cuts.main \
    --config-name "${config}" \
    "hydra.searchpath=[pkg://verl.trainer.config]" \
    "seed=${MC_SEED}" \
    "run_name=${MC_RUN_NAME}" \
    "paths.run_dir=${MC_RUN_DIR}" \
    "paths.model_dir=${MC_STAGED_MODEL_DIR}" \
    "paths.data_dir=${MC_STAGED_DATA_DIR}" \
    "paths.repo_dir=${MC_REPO_ROOT}" \
    "hydra.run.dir=${MC_JOB_DIR}/hydra" \
    "${extra[@]}" "$@"
  local rc=$?
  python "${MC_REPO_ROOT}/scripts/profile_memory.py" --report "${MC_RUN_DIR}" > "${MC_RUN_DIR}/memory_profile.md" 2>/dev/null || true
  return "${rc}"
}
