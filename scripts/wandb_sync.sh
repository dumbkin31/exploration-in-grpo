#!/usr/bin/env bash
# Sync offline W&B runs from Ada to wandb.ai, run from YOUR LAPTOP (not the Ada login node).
#
# Why: jobs log with WANDB_MODE=offline into the run's durable mirror on /home2 (<home>/mixed-cuts-data/runs/<run>/wandb/). `wandb sync` needs the
# wandb package, whose Linux wheels require glibc >= 2.28; the Ada login node (CentOS 7) has
# glibc 2.17, so the sync cannot run there. This script copies the offline run folders down with
# rsync (incremental, safe to repeat) and syncs them from here.
#
# Usage:
#   scripts/wandb_sync.sh <ssh-target> <ada-user> [run-name ...]
#   scripts/wandb_sync.sh ada lokola13                       # every run of that user
#   scripts/wandb_sync.sh ada lokola13 math_mixed_cuts-s1    # one run
#
# <ssh-target> is whatever you type after `ssh` (host alias or user@host).
# Needs locally: rsync, and `wandb` logged in (`pip install wandb && wandb login`). The
# entity/project come from each run's own metadata (WANDB_ENTITY/WANDB_PROJECT set on Ada),
# and the run id is the run name, so repeated syncs update the same W&B run.
set -euo pipefail

TARGET="${1:?usage: wandb_sync.sh <ssh-target> <ada-user> [run-name ...]}"
ADA_USER="${2:?usage: wandb_sync.sh <ssh-target> <ada-user> [run-name ...]}"
shift 2
REMOTE_RUNS="${MC_REMOTE_RUNS_DIR:-/home2/${ADA_USER}/mixed-cuts-data/runs}"   # MC_RUNS_DIR on Ada (docs/decisions/010)
LOCAL_ROOT="${MC_LOCAL_WANDB_DIR:-${HOME}/mixed-cuts-wandb}"

command -v rsync >/dev/null || { echo "rsync not found"; exit 1; }
command -v wandb >/dev/null || { echo "wandb not found: pip install wandb && wandb login"; exit 1; }

if [ "$#" -eq 0 ]; then
  RUNS=()  # no mapfile: macOS ships bash 3.2
  while IFS= read -r line; do [ -n "${line}" ] && RUNS+=("${line}"); done < <(ssh "${TARGET}" "ls -1 ${REMOTE_RUNS} 2>/dev/null")
else
  RUNS=("$@")
fi
[ "${#RUNS[@]}" -gt 0 ] || { echo "no runs found under ${TARGET}:${REMOTE_RUNS}"; exit 1; }

for run in "${RUNS[@]}"; do
  echo "== ${run}"
  dest="${LOCAL_ROOT}/${run}"
  mkdir -p "${dest}"
  if ! rsync -a "${TARGET}:${REMOTE_RUNS}/${run}/wandb/" "${dest}/"; then
    echo "   no wandb/ folder for ${run} (not started yet?); skipping"
    continue
  fi
  shopt -s nullglob
  offline=("${dest}"/offline-run-*)
  shopt -u nullglob
  if [ "${#offline[@]}" -eq 0 ]; then
    echo "   no offline runs in ${dest}; skipping"
    continue
  fi
  # --include-synced re-uploads runs that grew since the last sync (resumed jobs append to them).
  wandb sync --include-synced "${offline[@]}"
done
echo "done; local copies in ${LOCAL_ROOT}"
