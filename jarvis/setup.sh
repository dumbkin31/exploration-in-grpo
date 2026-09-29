#!/usr/bin/env bash
# One-time (idempotent) setup of a Jarvislabs GPU instance for the Mixed-CUTS runs (decision 013).
#
#   git clone <repo> /home/mixed-cuts && cd /home/mixed-cuts
#   cp /path/to/.env .env          # WANDB_API_KEY, WANDB_ENTITY=anlp-mixed-cuts, HF_TOKEN (GPQA is gated)
#   bash jarvis/setup.sh
#
# Steps: driver check (>= 580 for the CUDA 13 wheels) -> uv -> venv with the exact Ada lock (plus this
# package) -> model + datasets -> parquet data -> unit tests + config checks -> GPU preflight.
# Rerun it after an instance resume: finished steps are skipped (uv lives in /root, which a pause wipes).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=jarvis
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
say() { printf '\n== %s\n' "$*"; }

say "location"
case "${MC_REPO_ROOT}" in
  /home/*) echo "repo ${MC_REPO_ROOT} (persists across pause)";;
  *) echo "WARN: repo ${MC_REPO_ROOT} is not under /home: it and .venv are lost when the instance pauses";;
esac
echo "stage root ${MC_STAGE_ROOT} (model, data, runs, checkpoints)"

say "GPU and driver"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
if [ "${drv%%.*}" -lt "${MC_MIN_DRIVER_MAJOR}" ]; then
  echo "ERROR: driver ${drv} < ${MC_MIN_DRIVER_MAJOR}: the pinned torch 2.11 / vLLM 0.24 wheels are CUDA 13.0 builds."
  echo "       Pick a Jarvislabs image or instance with a >= ${MC_MIN_DRIVER_MAJOR} driver (see docs/decisions/013)."
  exit 2
fi

say "secrets"
if [ -f .env ]; then
  for k in WANDB_API_KEY WANDB_ENTITY HF_TOKEN; do
    if grep -q "^${k}=." .env; then echo "${k}: set"; else echo "WARN: ${k} missing from .env"; fi
  done
else
  echo "WARN: no .env: W&B logging (WANDB_API_KEY, WANDB_ENTITY=anlp-mixed-cuts) and GPQA (HF_TOKEN) need it"
fi

say "uv"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --quiet uv
fi
uv --version

say "python environment (${MC_VENV_DIR})"
if [ -x "${MC_VENV_DIR}/bin/python" ] && "${MC_VENV_DIR}/bin/python" -c "import vllm, verl, mixed_cuts" 2>/dev/null; then
  echo "already installed: $("${MC_VENV_DIR}/bin/python" -c 'import torch, vllm, verl; print("torch", torch.__version__, "vllm", vllm.__version__, "verl", verl.__version__)')"
else
  uv venv --python "${MC_PYTHON_VERSION}" "${MC_VENV_DIR}"
  lock="$(mktemp)"
  grep -v '^-e ' requirements/lock.txt > "${lock}"   # the lock's own editable line points at the Ada checkout
  uv pip install --python "${MC_VENV_DIR}/bin/python" -r "${lock}"
  uv pip install --python "${MC_VENV_DIR}/bin/python" --no-deps -e .
  rm -f "${lock}"
fi

say "model + datasets -> ${MC_STAGE_ROOT}"
"${MC_VENV_DIR}/bin/python" scripts/prefetch.py

say "parquet data -> ${MC_DATA_DIR}"
if [ -f "${MC_DATA_DIR}/MANIFEST.json" ]; then
  echo "already built"
else
  "${MC_VENV_DIR}/bin/python" scripts/prepare_data.py
fi

say "unit tests + config checks"
"${MC_VENV_DIR}/bin/python" -m pytest -q --ignore=tests/gpu -p no:cacheprovider
for c in smoke math_grpo math_mixed_cuts; do
  "${MC_VENV_DIR}/bin/python" scripts/compose_config.py "${c}" --check ${MC_TRAIN_OVERRIDES}
done

say "GPU preflight"
CUDA_VISIBLE_DEVICES=0 "${MC_VENV_DIR}/bin/python" scripts/check_env.py --config math_grpo

grep -q 'export MC_SITE=jarvis' ~/.bashrc 2>/dev/null || echo 'export MC_SITE=jarvis' >> ~/.bashrc
say "ready. Next: bash jarvis/smoke.sh, then bash jarvis/start.sh (docs/decisions/013)"
