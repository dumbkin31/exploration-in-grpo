#!/usr/bin/env bash
#SBATCH -J mc-train
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
# Real training run. Submit with:  make sbatch-train CONFIG=math_mixed_cuts SEED=1
# Resubmitting the same CONFIG+SEED resumes from the newest checkpoint in the durable run dir.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG="${MC_CONFIG:?set MC_CONFIG (e.g. sbatch --export=ALL,MC_CONFIG=math_mixed_cuts,MC_SEED=1 slurm/train.sbatch)}"
# exec (not a child bash): SLURM delivers --signal=B:SIGUSR1 to THIS shell's PID, and the
# stage-out traps are installed inside run_train.sh, so it must run in this process.
exec bash slurm/run_train.sh "${CONFIG}" "$@"
