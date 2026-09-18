#!/usr/bin/env bash
#SBATCH -J mc-smoke
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem-per-cpu=3G
#SBATCH --time=05:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# End-to-end plumbing test: ~20 MATH problems, 2 steps, n=4 (2 std + 2 CUTS), memory plan A.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MC_JOB_NAME=smoke
bash slurm/run_train.sh smoke "$@"
