#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Environment for Kaggle notebooks (decision 014): NVIDIA T4 16 GB (sm_75, the same hardware profile as
# Ada's 2080 Ti), 12-hour sessions, ~30 GPU-hours per account per week, no SLURM, W&B online.
# Reached through configs/ada.env.sh when MC_SITE=kaggle (exported by kaggle/*.sh).
#
# Storage: everything lives on the session's local disk (/tmp) and is gone when the session ends.
# A run survives between sessions on a private Hugging Face Hub repo (MC_HUB_REPO, scripts/hub_sync.py):
# kaggle/train.sh pulls it at the start and pushes every new checkpoint. Small outputs are copied to
# /kaggle/working/outputs, which Kaggle keeps with the notebook version. The repo and its .venv stay
# out of /kaggle/working on purpose (its 20 GB output limit).
# ---------------------------------------------------------------------------

_mc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$_mc_repo_root}"
export MC_SITE=kaggle
export MC_HW_PROFILE=sm75                 # T4 = cc 7.5: fp16, TRITON_ATTN, the SDPA patch (as on Ada)

# --- where -------------------------------------------------------------------
export MC_USER="${MC_USER:-${USER:-$(id -un)}}"
export MC_STAGE_ROOT="${MC_STAGE_ROOT:-/tmp/mc-data}"        # session-local: model, data, runs
export MC_SCRATCH_ROOT="${MC_SCRATCH_ROOT:-/tmp/mc-scratch}"
export MC_OUTPUT_DIR="${MC_OUTPUT_DIR:-/kaggle/working/outputs}"   # kept with the notebook version
export MC_VENV_DIR="${MC_VENV_DIR:-${MC_REPO_ROOT}/.venv}"
export MC_PYTHON_VERSION="${MC_PYTHON_VERSION:-3.12}"

# --- model + data + runs -------------------------------------------------------
export MC_MODEL_NAME="${MC_MODEL_NAME:-Qwen/Qwen3-1.7B}"
export MC_MODELS_DIR="${MC_MODELS_DIR:-${MC_STAGE_ROOT}/models}"
export MC_MODEL_DIR="${MC_MODEL_DIR:-${MC_MODELS_DIR}/$(basename "${MC_MODEL_NAME}")}"
export MC_DATA_DIR="${MC_DATA_DIR:-${MC_STAGE_ROOT}/data}"
export MC_RUNS_DIR="${MC_RUNS_DIR:-${MC_STAGE_ROOT}/runs}"
export MC_SCRATCH_RUNS_DIR="${MC_SCRATCH_RUNS_DIR:-${MC_SCRATCH_ROOT}/runs}"
export MC_CHECKPOINT_HOME=durable        # the run dir holds the checkpoints; hub_sync carries it over
export MC_MIRROR_INTERVAL="${MC_MIRROR_INTERVAL:-600}"
export MC_BAD_NODES_FILE="${MC_BAD_NODES_FILE:-${MC_STAGE_ROOT}/bad_nodes.txt}"
# Kaggle images have shipped NVIDIA drivers older than 580 (560 reported). kaggle/setup.sh then installs
# the CUDA 12.9 builds of the same torch/vLLM versions and records it; CUDA 12.x runs on drivers >= 525.
if [ "$(cat "${MC_VENV_DIR}/.mc_cuda_variant" 2>/dev/null)" = "cu129" ]; then
  export MC_MIN_DRIVER_MAJOR="${MC_MIN_DRIVER_MAJOR:-525}"
else
  export MC_MIN_DRIVER_MAJOR="${MC_MIN_DRIVER_MAJOR:-580}"
fi

# --- sessions and the Hub -------------------------------------------------------------------------
export MC_SESSION_HOURS="${MC_SESSION_HOURS:-12}"            # Kaggle's session limit
export MC_SESSION_MARGIN_MIN="${MC_SESSION_MARGIN_MIN:-30}"  # kept free at the end: final push, outputs
export MC_STEP_HOURS_GUESS="${MC_STEP_HOURS_GUESS:-3.5}"     # T4 step-time guess until a step is measured
export MC_HUB_KEEP="${MC_HUB_KEEP:-2}"                        # checkpoints kept on the Hub per run
# MC_HUB_REPO (e.g. <hf-user>/mixed-cuts-runs) and HF_TOKEN (write) come from .env (kaggle notebook cell)

# --- run naming ---------------------------------------------------------------------------------
export MC_RUN_TAG="${MC_RUN_TAG:-t4}"    # math_grpo-t4-s1: separate from the Ada and H100 runs

# --- caches ---------------------------------------------------------------------------------------
export MC_CACHE_ROOT="${MC_CACHE_ROOT:-${MC_SCRATCH_ROOT}/cache}"
export MC_ON_LOGIN_NODE=0
export HF_HOME="${HF_HOME:-${MC_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MC_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${MC_CACHE_ROOT}/torchinductor}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MC_CACHE_ROOT}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MC_CACHE_ROOT}/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/uv-python}"
export PATH="${HOME}/.local/bin:${PATH}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-mc}"
export TMPDIR="${TMPDIR:-${MC_CACHE_ROOT}/tmp}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${MC_CACHE_ROOT}/wandb-cache}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${PIP_CACHE_DIR}" \
         "${UV_CACHE_DIR}" "${RAY_TMPDIR}" "${TMPDIR}" "${WANDB_CACHE_DIR}" 2>/dev/null || true

# --- logging: W&B online (team from .env: WANDB_ENTITY=anlp-mixed-cuts) ---------------------------
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-mixed-cuts}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"

# --- layout: one T4 per run, the Ada LoRA plan (configs/train/layout/kaggle_t4.yaml) ---------------
export MC_LAYOUT=kaggle_t4
export MC_SLURM_GPUS=1                   # GPUs per run; the config checks compare it with the layout
export MC_TRAIN_OVERRIDES="layout=kaggle_t4"   # memory=plan_b_lora and hardware=sm75 are the defaults
export MC_REWARD_WORKERS="${MC_REWARD_WORKERS:-2}"   # 4 vCPUs per session
mc_sbatch_args() { echo "mc_sbatch_args: no SLURM on Kaggle; use kaggle/train.sh" >&2; return 1; }
mc_sbatch_exclude() { return 0; }

# --- runtime knobs (as on Ada: sm_75 / fp16) ------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
unset VLLM_USE_V2_MODEL_RUNNER

# --- secrets / local overrides -------------------------------------------------
[ -f "${MC_REPO_ROOT}/.env" ] && set -a && . "${MC_REPO_ROOT}/.env" && set +a
[ -f "${MC_REPO_ROOT}/configs/local.env.sh" ] && . "${MC_REPO_ROOT}/configs/local.env.sh"

unset _mc_repo_root
