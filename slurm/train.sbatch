#!/usr/bin/env bash
#SBATCH -J mc-train
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem-per-cpu=3G
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# Real training run. Submit with:  make sbatch-train CONFIG=math_mixed_cuts   (sets MC_CONFIG)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG="${MC_CONFIG:?set MC_CONFIG (e.g. sbatch --export=ALL,MC_CONFIG=math_mixed_cuts slurm/train.sbatch)}"
export MC_JOB_NAME="${CONFIG}"
bash slurm/run_train.sh "${CONFIG}" "$@"
