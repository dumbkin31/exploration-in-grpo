#!/usr/bin/env bash
#SBATCH -J mc-data
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=01:00:00
#SBATCH -o slurm-%x-%j.out
# Build the parquet files (verl schema, DAPO deduplicated, digit filter, eval sets) from the
# prefetched Hub snapshots. Needs the full venv (pandas/pyarrow), so it runs on a compute node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source configs/ada.env.sh
[ -f "${MC_STAGE_ROOT}/MANIFEST.json" ] || { echo "run 'make prefetch' on the login node first"; exit 1; }
"${MC_VENV_DIR}/bin/python" scripts/prepare_data.py
echo "== preflight with the staged model + data =="
mkdir -p "${MC_STAGE_ROOT}/jobs"
"${MC_VENV_DIR}/bin/python" scripts/check_env.py --facts-json "${MC_STAGE_ROOT}/jobs/data_${SLURM_JOB_ID}_env_facts.json" || true
