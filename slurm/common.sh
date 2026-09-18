#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared job plumbing for every slurm/*.sbatch. Source it, then call:
#
#   mc_job_init            paths, env, preflight (aborts early on FAIL)
#   mc_stage_in            model + parquet from $MC_STAGE_ROOT -> node-local /scratch
#   mc_run_with_traps CMD  run CMD, and on SIGUSR1/SIGTERM/EXIT copy the run dir out
#   mc_stage_out           rsync $MC_RUN_DIR -> $MC_DURABLE_DIR (idempotent, safe to call twice)
#
# The single most important line in here is the trap: a run whose only checkpoint lives on
# /scratch when the job is killed has lost the run. Jobs request --signal=B:SIGUSR1@<secs> so
# the trap fires <secs> before the hard kill.
# ---------------------------------------------------------------------------
set -euo pipefail

_MC_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$(cd "${_MC_COMMON_DIR}/.." && pwd)}"

mc_log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

mc_job_init() {
  # shellcheck source=/dev/null
  source "${MC_REPO_ROOT}/configs/ada.env.sh"
  export MC_JOB_ID="${SLURM_JOB_ID:-local-$$}"
  export MC_JOB_NAME="${MC_JOB_NAME:-${SLURM_JOB_NAME:-job}}"
  export MC_RUN_DIR="${MC_RUN_DIR:-${MC_SCRATCH_ROOT}/runs/${MC_JOB_NAME}-${MC_JOB_ID}}"
  export MC_DURABLE_DIR="${MC_DURABLE_DIR:-${MC_RUNS_DIR}/${MC_JOB_NAME}-${MC_JOB_ID}}"
  export MC_STAGED_MODEL_DIR="${MC_SCRATCH_ROOT}/stage/models/$(basename "${MC_MODEL_DIR}")"
  export MC_STAGED_DATA_DIR="${MC_SCRATCH_ROOT}/stage/data"
  export MC_ENV_FACTS_JSON="${MC_RUN_DIR}/env_facts.json"
  export WANDB_DIR="${MC_RUN_DIR}/wandb"
  mkdir -p "${MC_RUN_DIR}" "${WANDB_DIR}" "${MC_SCRATCH_ROOT}/stage"
  export PATH="${MC_VENV_DIR}/bin:${PATH}"
  export PYTHONUNBUFFERED=1

  mc_log "job ${MC_JOB_ID} (${MC_JOB_NAME}) on $(hostname); run dir ${MC_RUN_DIR}; durable ${MC_DURABLE_DIR}"
  mc_log "python: $(command -v python) ; GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
  { echo "job_id=${MC_JOB_ID}"; echo "node=$(hostname)"; echo "git=$(git -C "${MC_REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo ?)"; date; } > "${MC_RUN_DIR}/job_info.txt"

  # Preflight: pins, sm_75, storage, internet. Aborts with a clear message on FAIL.
  python "${MC_REPO_ROOT}/scripts/check_env.py" "${MC_PREFLIGHT_ARGS:---no-staged}" | tee "${MC_RUN_DIR}/preflight.log"
  # Offline mode if the node cannot reach the Hub (facts written by check_env).
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
}

mc_stage_in() {
  # Reads straight from $MC_STAGE_ROOT if it is visible on this node; otherwise stops with a
  # clear message (an ssh/rsync-from-master path is intentionally NOT built until we know the
  # mount situation for real; see README "Storage").
  if [ ! -d "${MC_STAGE_ROOT}" ]; then
    mc_log "ERROR: ${MC_STAGE_ROOT} is not visible on $(hostname). Export MC_STAGE_ROOT to a path this node can read (or ask the admins whether /share1 is mounted on compute nodes)."
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

mc_stage_out() {
  # Idempotent: rsync only copies what changed, so calling it from the signal trap AND at exit
  # is cheap and completes a partial copy.
  mkdir -p "${MC_DURABLE_DIR}" 2>/dev/null || { mc_log "WARN: cannot create ${MC_DURABLE_DIR}; outputs stay in ${MC_RUN_DIR}"; return 0; }
  mc_log "stage-out ${MC_RUN_DIR} -> ${MC_DURABLE_DIR}"
  rsync -a --info=stats1 --exclude 'cuts_stats/*.tmp' "${MC_RUN_DIR}/" "${MC_DURABLE_DIR}/" || mc_log "WARN: stage-out rsync returned $?"
  mc_log "stage-out done"
}

_MC_CHILD_PID=""
_mc_on_signal() {
  local sig="$1"
  mc_log "caught ${sig}: staging out now (child pid ${_MC_CHILD_PID:-none})"
  mc_stage_out
  if [ -n "${_MC_CHILD_PID}" ] && kill -0 "${_MC_CHILD_PID}" 2>/dev/null; then
    kill -TERM "${_MC_CHILD_PID}" 2>/dev/null || true
    # give verl/ray a moment to flush, then a final copy
    for _ in $(seq 1 60); do kill -0 "${_MC_CHILD_PID}" 2>/dev/null || break; sleep 2; done
    mc_stage_out
  fi
  exit 143
}

mc_run_with_traps() {
  # Usage: mc_run_with_traps <command...>
  trap '_mc_on_signal SIGUSR1' USR1
  trap '_mc_on_signal SIGTERM' TERM
  trap 'mc_stage_out' EXIT
  mc_log "launch: $*"
  "$@" &
  _MC_CHILD_PID=$!
  local rc=0
  wait "${_MC_CHILD_PID}" || rc=$?
  _MC_CHILD_PID=""
  mc_log "command finished with rc=${rc}"
  return "${rc}"
}
