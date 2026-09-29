#!/usr/bin/env bash
# Train ONE arm on ONE T4 inside a Kaggle session, carrying the run across sessions (decision 014).
#
#   bash kaggle/train.sh <config> <seed> [gpu]      # e.g. bash kaggle/train.sh math_mixed_cuts 1 0
#
# 1. pulls the run's newest checkpoint from the Hub (MC_HUB_REPO), if any
# 2. trains (resumes from that checkpoint) and pushes every new checkpoint to the Hub
# 3. after each logged step: if the next step would not finish before the session limit
#    (MC_SESSION_HOURS minus MC_SESSION_MARGIN_MIN), stops cleanly (SIGUSR1: the step just logged is
#    checkpointed), pushes, and copies the small outputs to /kaggle/working/outputs/<run>
# Run the notebook again (a new session) to continue. Run name <config>-t4-s<seed> = the W&B run id.
set -uo pipefail
CONFIG="${1:?usage: kaggle/train.sh <config> <seed> [gpu]}"
SEED="${2:?usage: kaggle/train.sh <config> <seed> [gpu]}"
GPU="${3:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=kaggle
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
export CUDA_VISIBLE_DEVICES="${GPU}" MC_SEED="${SEED}"
export MC_RUN_NAME="${MC_RUN_NAME:-${CONFIG}-${MC_RUN_TAG}-s${SEED}}"
export RAY_TMPDIR="/tmp/ray-g${GPU}"
PY="${MC_VENV_DIR}/bin/python"
RUN_DIR="${MC_RUNS_DIR}/${MC_RUN_NAME}"
LOG_DIR="${RUN_DIR}/launcher"
mkdir -p "${LOG_DIR}" "${RAY_TMPDIR}"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_DIR}/launcher.log"; }
hub() { if [ -n "${MC_HUB_REPO:-}" ]; then "${PY}" scripts/hub_sync.py "$@"; else return 0; fi; }
tracker() { cat "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
last_logged() { tail -n 1 "${RUN_DIR}/metrics.jsonl" 2>/dev/null | sed -n 's/.*"step": *\([0-9]*\).*/\1/p'; }
elapsed() {    # seconds since the session started
  if [ -n "${MC_SESSION_START_EPOCH:-}" ]; then echo $(( $(date +%s) - MC_SESSION_START_EPOCH )); return; fi
  ps -o etimes= -p 1 2>/dev/null | tr -d ' ' || echo 0
}
step_estimate() {   # seconds: mean of the last 5 logged step times, else the guess
  "${PY}" - "${RUN_DIR}/metrics.jsonl" "${MC_STEP_HOURS_GUESS}" <<'PY'
import json, sys
t = []
try:
    for line in open(sys.argv[1]):
        try:
            v = json.loads(line).get("timing_s/step")
        except ValueError:
            continue
        if isinstance(v, (int, float)):
            t.append(v)
except OSError:
    pass
print(int(sum(t[-5:]) / len(t[-5:])) if t else int(float(sys.argv[2]) * 3600))
PY
}
save_outputs() {
  local out="${MC_OUTPUT_DIR}/${MC_RUN_NAME}"
  mkdir -p "${out}" 2>/dev/null || return 0
  cp -f "${RUN_DIR}"/{metrics.jsonl,phases.jsonl,memory_profile.md} "${out}/" 2>/dev/null || true
  cp -f "${LOG_DIR}"/*.log "${out}/" 2>/dev/null || true
}

[ -n "${MC_HUB_REPO:-}" ] || log "WARN: MC_HUB_REPO unset: nothing carries this run to the next session"
hub pull --run "${MC_RUN_NAME}" | tee -a "${LOG_DIR}/launcher.log" || log "WARN: pull failed"
total="$("${PY}" scripts/compose_config.py "${CONFIG}" ${MC_TRAIN_OVERRIDES} --select trainer.total_training_steps 2>/dev/null | head -1 | tr -d ' ')"
[[ "${total}" =~ ^[0-9]+$ ]] || total=100
[ -n "${MC_TOTAL_STEPS:-}" ] && total="${MC_TOTAL_STEPS}"
limit=$(awk -v h="${MC_SESSION_HOURS}" -v m="${MC_SESSION_MARGIN_MIN}" 'BEGIN{print int(h*3600 - m*60)}')
if [ "$(tracker)" -ge "${total}" ]; then log "already finished (step ${total})"; save_outputs; exit 0; fi

export MC_JOB_ID="t4-$(date +%Y%m%d-%H%M%S)-g${GPU}"
export MC_JOB_STDOUT="${LOG_DIR}/${MC_JOB_ID}.out"
start_step="$(tracker)"
log "run ${MC_RUN_NAME} on GPU ${GPU}: resuming after step ${start_step} of ${total}; session budget $((limit / 60)) min, $(( $(elapsed) / 60 )) min used"
bash slurm/run_train.sh "${CONFIG}" > "${MC_JOB_STDOUT}" 2>&1 &
pid=$!
pushed="${start_step}" seen="$(last_logged)" stopping=0
while kill -0 "${pid}" 2>/dev/null; do
  sleep "${MC_POLL_SECS:-60}"
  if [ "$(tracker)" != "${pushed}" ]; then
    hub push --run "${MC_RUN_NAME}" >> "${LOG_DIR}/launcher.log" 2>&1 && pushed="$(tracker)" && log "pushed checkpoint ${pushed}"
  fi
  now="$(last_logged)"
  if [ "${stopping}" = 0 ] && [ -n "${now}" ] && [ "${now}" != "${seen}" ]; then
    seen="${now}"
    left=$(( limit - $(elapsed) )); need=$(( ($(step_estimate) * 11 + 9) / 10 ))
    log "step ${now} logged; ${left}s left in this session, next step needs ~${need}s"
    if [ "${now}" -lt "${total}" ] && [ "${left}" -lt "${need}" ]; then
      log "stopping after step ${now}: the next step would not finish before the session limit"
      stopping=1
      kill -USR1 "${pid}" 2>/dev/null || true
    fi
  fi
done
wait "${pid}"; rc=$?
hub push --run "${MC_RUN_NAME}" >> "${LOG_DIR}/launcher.log" 2>&1 && log "pushed checkpoint $(tracker)"
save_outputs
if [ "$(tracker)" -ge "${total}" ]; then log "finished: step ${total}"; exit 0; fi
if [ "${stopping}" = 1 ]; then
  log "session done at step $(tracker) of ${total} (this session: +$(( $(tracker) - start_step ))). Run the notebook again to continue."
  exit 0
fi
log "training stopped unexpectedly (rc=${rc}) at step $(tracker); tail of ${MC_JOB_STDOUT}:"
tail -n 30 "${MC_JOB_STDOUT}" | tee -a "${LOG_DIR}/launcher.log"
exit "${rc}"
