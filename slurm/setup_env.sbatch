#!/usr/bin/env bash
#SBATCH -J mc-setup
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:00:00
#SBATCH -o slurm-%x-%j.out
# Build the full cluster venv ON A 2080 TI NODE (glibc 2.35), then freeze it. Never on the login node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source configs/ada.env.sh
echo "building ${MC_VENV_DIR} on $(hostname)"
command -v uv >/dev/null || { echo "uv not found; install it in your home with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
make setup
make lock
"${MC_VENV_DIR}/bin/python" scripts/check_env.py --no-staged || true
"${MC_VENV_DIR}/bin/python" -m pytest -q tests/test_vllm_logits_processor.py tests/test_agent_loop_verl.py
make compose-check
if [ -f "${MC_STAGE_ROOT}/MANIFEST.json" ]; then make data; else echo "skip 'make data': run 'make prefetch' on the login node first"; fi
echo "done; commit requirements/lock.txt"
