#!/usr/bin/env bash
#SBATCH -J mc-train
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# Real training run. Submit with:  make sbatch-train CONFIG=math_mixed_cuts SEED=1
# Resubmitting the same CONFIG+SEED resumes from the newest checkpoint (on the node named in node.txt).
# --time is the QoS low MaxWall (4 days); a 100-step run at ~1-1.5 h/step needs two submissions (012).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG="${MC_CONFIG:?set MC_CONFIG (e.g. sbatch --export=ALL,MC_CONFIG=math_mixed_cuts,MC_SEED=1 slurm/train.sbatch)}"
# exec (not a child bash): SLURM delivers --signal=B:SIGUSR1 to THIS shell's PID, and the
# stage-out traps are installed inside run_train.sh, so it must run in this process.
exec bash slurm/run_train.sh "${CONFIG}" "$@"
