#!/usr/bin/env bash
# Launch the arms in the background, ONE PER GPU (decision 013):
#   2-GPU instance:              bash jarvis/start.sh               -> math_grpo on GPU 0, math_mixed_cuts on GPU 1
#   two 1-GPU instances:         ARMS=math_grpo bash jarvis/start.sh      (on the first)
#                                ARMS=math_mixed_cuts bash jarvis/start.sh (on the second)
# SEED (default 1) picks the seed. Already-running arms are left alone, so rerunning after a pause or a
# spot preemption restarts exactly the arms that stopped; each resumes from its newest checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=jarvis
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
SEED="${SEED:-1}"
read -r -a arms <<< "${ARMS:-math_grpo math_mixed_cuts}"
ngpu="$(nvidia-smi -L | grep -c '^GPU')"
if [ "${#arms[@]}" -gt "${ngpu}" ]; then
  echo "ERROR: ${#arms[@]} arms but ${ngpu} GPU(s). One arm per GPU: ARMS=math_grpo bash jarvis/start.sh" >&2
  exit 2
fi
for i in "${!arms[@]}"; do
  arm="${arms[$i]}"
  run="${arm}-${MC_RUN_TAG}-s${SEED}"
  if pgrep -f "jarvis/train.sh ${arm} ${SEED} " >/dev/null; then
    echo "${run}: already running"
    continue
  fi
  mkdir -p "${MC_RUNS_DIR}/${run}/launcher"
  nohup setsid bash jarvis/train.sh "${arm}" "${SEED}" "${i}" \
    >> "${MC_RUNS_DIR}/${run}/launcher/nohup.out" 2>&1 < /dev/null &
  echo "${run}: started on GPU ${i} (pid $!); follow with: bash jarvis/status.sh"
done
