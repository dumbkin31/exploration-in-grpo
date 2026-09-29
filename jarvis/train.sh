#!/usr/bin/env bash
# Run ONE arm on ONE GPU until it reaches its last step (decision 013). Resumes from the newest
# checkpoint in the run dir, restarts after a crash (up to MC_MAX_ATTEMPTS), and exits 0 once done.
# Rerunning the same command after an instance pause or spot preemption continues the run.
#
#   bash jarvis/train.sh <config> <seed> [gpu]      # e.g. bash jarvis/train.sh math_grpo 1 0
#
# Run name <config>-h100-s<seed> is also the W&B run id (team from .env, project mixed-cuts).
# jarvis/start.sh launches the arms in the background with this script.
set -uo pipefail
CONFIG="${1:?usage: jarvis/train.sh <config> <seed> [gpu]}"
SEED="${2:?usage: jarvis/train.sh <config> <seed> [gpu]}"
GPU="${3:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=jarvis
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
export CUDA_VISIBLE_DEVICES="${GPU}" MC_SEED="${SEED}"
export MC_RUN_NAME="${MC_RUN_NAME:-${CONFIG}-${MC_RUN_TAG}-s${SEED}}"
export RAY_TMPDIR="/tmp/ray-g${GPU}"     # one Ray instance per GPU; a short path (Ray's socket-path limit)
mkdir -p "${RAY_TMPDIR}"
RUN_DIR="${MC_RUNS_DIR}/${MC_RUN_NAME}"
LOG_DIR="${RUN_DIR}/launcher"
mkdir -p "${LOG_DIR}"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_DIR}/launcher.log"; }

total="$("${MC_VENV_DIR}/bin/python" scripts/compose_config.py "${CONFIG}" ${MC_TRAIN_OVERRIDES} \
          --select trainer.total_training_steps 2>/dev/null | head -1 | tr -d ' ')"
[[ "${total}" =~ ^[0-9]+$ ]] || total=100
[ -n "${MC_TOTAL_STEPS:-}" ] && total="${MC_TOTAL_STEPS}"

finished() {   # the last step's checkpoint exists (verl saves at the final step regardless of save_freq)
  local f="${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt"
  [ -f "${f}" ] && [ "$(cat "${f}")" -ge "${total}" ] 2>/dev/null
}

gpu_free() {   # wait up to 5 min for this GPU to be released by a previous attempt
  for _ in $(seq 1 30); do
    used="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    [ -n "${used}" ] && [ "${used}" -lt 1000 ] && return 0
    sleep 10
  done
  return 1
}

log "run ${MC_RUN_NAME} on GPU ${GPU}: ${total} steps, layout ${MC_LAYOUT} (${MC_TRAIN_OVERRIDES})"
max="${MC_MAX_ATTEMPTS:-6}"
for attempt in $(seq 1 "${max}"); do
  if finished; then log "already finished (checkpoint at step ${total})"; exit 0; fi
  export MC_JOB_ID="h100-$(date +%Y%m%d-%H%M%S)-g${GPU}"
  export MC_JOB_STDOUT="${LOG_DIR}/${MC_JOB_ID}.out"
  last="$(cat "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
  log "attempt ${attempt}/${max}: ${MC_JOB_ID} (resuming after checkpoint: ${last}); log ${MC_JOB_STDOUT}"
  bash slurm/run_train.sh "${CONFIG}" > "${MC_JOB_STDOUT}" 2>&1
  rc=$?
  if finished; then
    log "finished: step ${total} checkpointed (rc=${rc})"
    if [ -n "${MC_HUB_REPO:-}" ]; then   # optional copy on the Hub (e.g. to evaluate it on Kaggle, decision 014)
      "${MC_VENV_DIR}/bin/python" scripts/hub_sync.py push --run "${MC_RUN_NAME}" >> "${LOG_DIR}/launcher.log" 2>&1 \
        && log "pushed to ${MC_HUB_REPO}" || log "WARN: hub push failed"
    fi
    exit 0
  fi
  log "attempt ${attempt} ended with rc=${rc} before step ${total}; tail of its log:"
  tail -n 15 "${MC_JOB_STDOUT}" | tee -a "${LOG_DIR}/launcher.log"
  # leftovers of THIS run's Ray instance only (the other arm has its own RAY_TMPDIR)
  pkill -f "${RAY_TMPDIR}/" 2>/dev/null || true
  gpu_free || log "WARN: GPU ${GPU} still busy after 5 min"
  sleep 30
done
log "gave up after ${max} attempts; fix the cause, then rerun: bash jarvis/train.sh ${CONFIG} ${SEED} ${GPU}"
exit 1
