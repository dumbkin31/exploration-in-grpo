#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Environment for Jarvislabs (decision 013): one NVIDIA H100 80 GB per run, no SLURM, W&B online.
# Reached through configs/ada.env.sh when MC_SITE=jarvis (every script sources that file), so the
# Makefile targets, slurm/run_train.sh and the checks work unchanged:
#
#   export MC_SITE=jarvis; source configs/ada.env.sh
#
# Storage on Jarvislabs: only /home survives a pause; a persistent filesystem volume survives even
# instance deletion. Keep the repo (and its .venv) under /home; point MC_STAGE_ROOT at the volume if
# one is attached (e.g. in configs/local.env.sh). Checkpoints live in the durable run dir.
# ---------------------------------------------------------------------------

_mc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$_mc_repo_root}"
export MC_SITE=jarvis
export MC_HW_PROFILE=h100

# --- who / where -------------------------------------------------------------
export MC_USER="${MC_USER:-${USER:-$(id -un)}}"
export MC_STAGE_ROOT="${MC_STAGE_ROOT:-/home/mixed-cuts-data}"              # persistent: model, data, runs
export MC_SCRATCH_ROOT="${MC_SCRATCH_ROOT:-/root/mc-scratch}"             # instance-local working copies
export MC_VENV_DIR="${MC_VENV_DIR:-${MC_REPO_ROOT}/.venv}"               # under /home with the repo
export MC_PYTHON_VERSION="${MC_PYTHON_VERSION:-3.12}"

# --- model + data + runs -------------------------------------------------------
export MC_MODEL_NAME="${MC_MODEL_NAME:-Qwen/Qwen3-1.7B}"
export MC_MODELS_DIR="${MC_MODELS_DIR:-${MC_STAGE_ROOT}/models}"
export MC_MODEL_DIR="${MC_MODEL_DIR:-${MC_MODELS_DIR}/$(basename "${MC_MODEL_NAME}")}"
export MC_DATA_DIR="${MC_DATA_DIR:-${MC_STAGE_ROOT}/data}"
export MC_RUNS_DIR="${MC_RUNS_DIR:-${MC_STAGE_ROOT}/runs}"
export MC_SCRATCH_RUNS_DIR="${MC_SCRATCH_RUNS_DIR:-${MC_SCRATCH_ROOT}/runs}"
export MC_CHECKPOINT_HOME=durable        # 0.9 GB LoRA checkpoints next to the metrics: a resume after a
                                         # pause or a spot preemption finds them on the persistent disk
export MC_MIRROR_INTERVAL="${MC_MIRROR_INTERVAL:-600}"
export MC_BAD_NODES_FILE="${MC_BAD_NODES_FILE:-${MC_STAGE_ROOT}/bad_nodes.txt}"
export MC_MIN_DRIVER_MAJOR="${MC_MIN_DRIVER_MAJOR:-580}"   # the pinned torch/vLLM wheels are CUDA 13.0 builds

# --- run naming: the H100 runs are new W&B runs, separate from the Ada attempts -------------------
export MC_RUN_TAG="${MC_RUN_TAG:-h100}"  # run name <config>-h100-s<seed> (slurm/common.sh::mc_train_main)

# --- caches (instance-local; rebuilt when missing) -----------------------------------------------
export MC_CACHE_ROOT="${MC_CACHE_ROOT:-${MC_SCRATCH_ROOT}/cache}"
export MC_ON_LOGIN_NODE=0
export HF_HOME="${HF_HOME:-${MC_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MC_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${MC_CACHE_ROOT}/torchinductor}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MC_CACHE_ROOT}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MC_CACHE_ROOT}/uv}"
# uv-managed interpreters must survive a pause (the venv symlinks to them): keep them under /home
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/home/.uv-python}"
export PATH="${HOME}/.local/bin:${PATH}"
# Ray: one instance per run (two runs share a 2-GPU machine); jarvis/train.sh sets RAY_TMPDIR per run.
# Keep the path short: Ray's unix sockets live under it (108-byte limit).
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-mc}"
export TMPDIR="${TMPDIR:-${MC_CACHE_ROOT}/tmp}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${MC_CACHE_ROOT}/wandb-cache}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${PIP_CACHE_DIR}" \
         "${UV_CACHE_DIR}" "${RAY_TMPDIR}" "${TMPDIR}" "${WANDB_CACHE_DIR}" 2>/dev/null || true

# --- logging: W&B online (team from .env: WANDB_ENTITY=anlp-mixed-cuts) ---------------------------
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-mixed-cuts}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"

# --- layout: one H100 per run (configs/train/{layout,memory,hardware}/) --------------------------
export MC_LAYOUT=h100_1gpu
export MC_SLURM_GPUS=1                   # GPUs per run; the config checks compare it with the layout
export MC_TRAIN_OVERRIDES="layout=h100_1gpu memory=plan_b_lora_gpu hardware=h100"
export MC_REWARD_WORKERS="${MC_REWARD_WORKERS:-4}"
mc_sbatch_args() { echo "mc_sbatch_args: no SLURM on Jarvislabs; use jarvis/train.sh" >&2; return 1; }
mc_sbatch_exclude() { return 0; }

# --- runtime knobs -------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # same sampler path as the Ada runs
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
unset VLLM_USE_V2_MODEL_RUNNER            # custom logits processors need Model Runner V1

# --- secrets / local overrides -------------------------------------------------
[ -f "${MC_REPO_ROOT}/.env" ] && set -a && . "${MC_REPO_ROOT}/.env" && set +a
[ -f "${MC_REPO_ROOT}/configs/local.env.sh" ] && . "${MC_REPO_ROOT}/configs/local.env.sh"

unset _mc_repo_root
