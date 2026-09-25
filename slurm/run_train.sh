#!/usr/bin/env bash
# In-allocation training launcher, used by `make train/smoke` and exec'd by the sbatch scripts.
#
#   bash slurm/run_train.sh <config-name> [extra hydra overrides...]
#   MC_SEED=1 bash slurm/run_train.sh math_mixed_cuts          # run name math_mixed_cuts-s1
#
# Everything (paths, seed, run name, resume, traps) is in slurm/common.sh::mc_train_main.
set -euo pipefail
CONFIG="${1:?usage: run_train.sh <config-name> [overrides...]}"; shift || true
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mc_train_main "${CONFIG}" "$@"
