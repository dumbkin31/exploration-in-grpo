#!/usr/bin/env bash
#SBATCH -J mc-setup
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=03:00:00
#SBATCH -o slurm-%x-%j.out
# Build the full cluster venv ON A 2080 TI NODE (Ubuntu 22.04, glibc 2.35), then freeze it, run the
# vLLM/verl interface tests, compose-check the configs and build the parquet data. Never on the
# login node (CentOS 7). Does not depend on `make` (not guaranteed on compute nodes).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source configs/ada.env.sh
VERL_TAG="${VERL_TAG:-v0.9.0}"
PY="${MC_VENV_DIR}/bin/python"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "node $(hostname): $(head -1 /etc/os-release) glibc $(ldd --version | head -1 | awk '{print $NF}')"
log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
log "scratch: $(df -h /scratch 2>/dev/null | tail -1 || echo 'NO /scratch on this node')"
log "internet: huggingface.co $(curl -s -m 8 -o /dev/null -w '%{http_code}' https://huggingface.co || echo unreachable), pypi.org $(curl -s -m 8 -o /dev/null -w '%{http_code}' https://pypi.org || echo unreachable)"
command -v uv >/dev/null || { log "uv not found; install it on the login node: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
log "uv $(uv --version) ; venv -> ${MC_VENV_DIR} ; python cache -> ${UV_PYTHON_INSTALL_DIR}"

log "== 1/6 venv + pinned requirements (requirements/base.txt)"
uv venv --python "${MC_PYTHON_VERSION}" "${MC_VENV_DIR}"
uv pip install --python "${PY}" -r requirements/base.txt
log "== 2/6 verl ${VERL_TAG} (no deps) + this package"
uv pip install --python "${PY}" --no-deps "verl @ git+https://github.com/volcengine/verl.git@${VERL_TAG}"
uv pip install --python "${PY}" --no-deps -e .
log "== 3/6 freeze -> requirements/lock.txt"
uv pip freeze --python "${PY}" > requirements/lock.txt
log "$(wc -l < requirements/lock.txt) pinned lines"
log "== 4/6 preflight (no staged-data check yet)"
"${PY}" scripts/check_env.py --no-staged --facts-json "${MC_STAGE_ROOT}/setup_env_facts_${SLURM_JOB_ID}.json" || true
log "== 5/6 interface tests that need vLLM/verl + compose-check"
"${PY}" -m pytest -q tests/test_vllm_logits_processor.py tests/test_agent_loop_verl.py -p no:cacheprovider
for c in base_grpo smoke math_grpo math_mixed_cuts; do "${PY}" scripts/compose_config.py "$c" --check; done
"${PY}" scripts/compose_config.py math_mixed_cuts --check memory=plan_b_lora
log "== 6/6 parquet data"
if [ -f "${MC_STAGE_ROOT}/MANIFEST.json" ]; then
  "${PY}" scripts/prepare_data.py
else
  log "skip prepare_data: ${MC_STAGE_ROOT}/MANIFEST.json missing (prefetch not finished); run it later with: sbatch slurm/data.sbatch"
fi
log "done; commit requirements/lock.txt"
