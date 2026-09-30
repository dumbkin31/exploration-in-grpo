#!/usr/bin/env bash
# Session setup in a Kaggle notebook (decision 014). Run at the start of EVERY session: Kaggle keeps
# nothing on disk between sessions. ~10-15 minutes (downloads). The notebook kaggle/mixed_cuts_kaggle.ipynb
# clones the repo to /tmp/mixed-cuts, writes .env from Kaggle Secrets and calls this script.
#
# Steps: GPUs + driver -> uv -> venv (the exact Ada lock on drivers >= 580; otherwise the CUDA 12.9 builds
# of the same torch/vLLM with every other package pinned to the lock) -> model + datasets -> parquet ->
# unit tests + config checks -> GPU preflight.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=kaggle
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
say() { printf '\n== %s\n' "$*"; }
T0=$(date +%s)

say "GPUs and driver"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "P100"; then
  echo "ERROR: P100 (cc 6.0) is not supported by CUDA 13 / vLLM 0.24. Set the accelerator to 'GPU T4 x2'."
  exit 2
fi
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
if [ "${drv%%.*}" -ge 580 ]; then variant=cu130; else variant=cu129; fi
echo "driver ${drv} -> torch/vLLM CUDA build: ${variant}"
free -g | head -2

say "secrets"
if [ -f .env ]; then
  for k in WANDB_API_KEY WANDB_ENTITY HF_TOKEN MC_HUB_REPO; do
    if grep -q "^${k}=." .env; then echo "${k}: set"; else echo "WARN: ${k} missing from .env"; fi
  done
else
  echo "WARN: no .env (the notebook writes it from Kaggle Secrets)"
fi

say "uv"
command -v uv >/dev/null 2>&1 || python3 -m pip install --user --quiet uv
uv --version

say "python environment (${MC_VENV_DIR}, ${variant})"
if [ -x "${MC_VENV_DIR}/bin/python" ] && [ "$(cat "${MC_VENV_DIR}/.mc_cuda_variant" 2>/dev/null)" = "${variant}" ] \
   && "${MC_VENV_DIR}/bin/python" -c "import vllm, verl, mixed_cuts" 2>/dev/null; then
  echo "already installed"
else
  uv venv --python "${MC_PYTHON_VERSION}" "${MC_VENV_DIR}"
  PY="${MC_VENV_DIR}/bin/python"
  tmp="$(mktemp -d)"
  verl_req="$(grep '^verl @ ' requirements/lock.txt)"
  if [ "${variant}" = cu130 ]; then
    grep -v -e '^-e ' requirements/lock.txt > "${tmp}/lock.txt"   # the lock's editable line points at Ada
    uv pip install --python "${PY}" -r "${tmp}/lock.txt"
  else
    # every non-CUDA package pinned to the lock; torch/vLLM and the CUDA runtime packages resolved for 12.9
    grep -v -E '^(-e |verl @ |torch==|torchvision==|torchaudio==|vllm==|nvidia-|cuda-|flashinfer)' \
      requirements/lock.txt > "${tmp}/constraints.txt"
    uv pip install --python "${PY}" -c "${tmp}/constraints.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
      "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129" \
      "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      -r requirements/base.txt
    uv pip install --python "${PY}" --no-deps "${verl_req}"
  fi
  uv pip install --python "${PY}" --no-deps -e .
  echo "${variant}" > "${MC_VENV_DIR}/.mc_cuda_variant"
  rm -rf "${tmp}"
  source configs/ada.env.sh   # re-read: the minimum driver depends on the variant just installed
fi
"${MC_VENV_DIR}/bin/python" -c 'import torch, vllm, verl; print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| verl", verl.__version__)'

say "model + datasets -> ${MC_STAGE_ROOT}"
"${MC_VENV_DIR}/bin/python" scripts/prefetch.py

say "parquet data -> ${MC_DATA_DIR}"
[ -f "${MC_DATA_DIR}/MANIFEST.json" ] && echo "already built" || "${MC_VENV_DIR}/bin/python" scripts/prepare_data.py

say "unit tests + config checks"
"${MC_VENV_DIR}/bin/python" -m pytest -q --ignore=tests/gpu --ignore=tests/test_trainer_helpers.py -p no:cacheprovider
for c in smoke math_grpo math_mixed_cuts; do
  "${MC_VENV_DIR}/bin/python" scripts/compose_config.py "${c}" --check ${MC_TRAIN_OVERRIDES}
done

say "GPU preflight (GPU 0)"
CUDA_VISIBLE_DEVICES=0 "${MC_VENV_DIR}/bin/python" scripts/check_env.py --config math_grpo

say "ready after $(( ($(date +%s) - T0) / 60 )) min. Next: kaggle/smoke.sh, kaggle/train.sh or kaggle/eval.sh"
