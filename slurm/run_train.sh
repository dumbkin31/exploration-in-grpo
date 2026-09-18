#!/usr/bin/env bash
# In-allocation training launcher (used by `make train`, `make smoke` and the sbatch scripts).
#
#   bash slurm/run_train.sh <config-name> [extra hydra overrides...]
#
# Resolves every path from configs/ada.env.sh + slurm/common.sh, stages data in, and runs
# python -m mixed_cuts.main with the config from configs/train/<config-name>.yaml.
set -euo pipefail
CONFIG="${1:?usage: run_train.sh <config-name> [overrides...]}"; shift || true
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export MC_JOB_NAME="${MC_JOB_NAME:-${CONFIG}}"
mc_job_init
mc_stage_in

export MC_EXPERIMENT_NAME="${MC_EXPERIMENT_NAME:-${CONFIG}-${MC_JOB_ID}}"
cd "${MC_REPO_ROOT}"
mc_run_with_traps python -m mixed_cuts.main \
  --config-name "${CONFIG}" \
  "hydra.searchpath=[pkg://verl.trainer.config]" \
  "paths.run_dir=${MC_RUN_DIR}" \
  "paths.model_dir=${MC_STAGED_MODEL_DIR}" \
  "paths.data_dir=${MC_STAGED_DATA_DIR}" \
  "paths.repo_dir=${MC_REPO_ROOT}" \
  "trainer.experiment_name=${MC_EXPERIMENT_NAME}" \
  "hydra.run.dir=${MC_RUN_DIR}/hydra" \
  "$@"
