#!/usr/bin/env bash
#SBATCH -J mc-resume-test
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
# Checkpoint-resume verification (Task C): run the smoke config for 30 steps with save_freq 5,
# kill the trainer through the real SIGUSR1 trap once step 15 is logged, then resubmit this
# script with a dependency; the second job must resume at step 16 with the same W&B run id.
#   sbatch slurm/test_resume.sbatch            # phase 1 (kills itself at step 15, resubmits)
#   ...the second job runs to step 30 and checks the invariants with scripts/check_resume.py
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MC_SEED="${MC_SEED:-7}"
export MC_RUN_NAME="${MC_RUN_NAME:-resume-test-s${MC_SEED}}"   # SAME dir for both phases
export MC_SAVE_FREQ=5 MC_TOTAL_STEPS=30
PHASE="${MC_RESUME_PHASE:-1}"
if [ "${PHASE}" = "1" ]; then
  export MC_KILL_AFTER_STEP=15
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --export=ALL,MC_RESUME_PHASE=2,MC_KILL_AFTER_STEP= slurm/test_resume.sbatch
  exec bash slurm/run_train.sh smoke "$@"
else
  unset MC_KILL_AFTER_STEP
  bash slurm/run_train.sh smoke "$@" || true
  source configs/ada.env.sh
  "${MC_VENV_DIR}/bin/python" scripts/check_resume.py "${MC_RUNS_DIR}/${MC_RUN_NAME}" --killed-at 15 --final-step 30 --save-freq 5
fi
