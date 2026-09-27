#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Single source of truth for every cluster path and environment variable.
# Sourced by the Makefile, every slurm/*.sbatch and scripts/check_env.py (via env).
#
#   source configs/ada.env.sh
#
# Rules (see README "Storage"):
#   /home2/$USER   25 GB NFS quota   -> code + venv ONLY
#   /share1        durable           -> staged model/datasets, run outputs after a job
#   /scratch       node-local, purged -> working copies, caches, live run dir
# Nothing in src/ may hardcode a cluster path; it all comes from here.
# Override any variable by exporting it BEFORE sourcing this file, or by putting
# exports in configs/local.env.sh (git-ignored), which is sourced last.
# ---------------------------------------------------------------------------

_mc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$_mc_repo_root}"

# --- who / where -------------------------------------------------------------
export MC_USER="${MC_USER:-${USER:-$(id -un)}}"
export MC_STAGE_ROOT="${MC_STAGE_ROOT:-/share1/${MC_USER}/mixed-cuts}"      # durable
export MC_SCRATCH_ROOT="${MC_SCRATCH_ROOT:-/scratch/${MC_USER}/mixed-cuts}" # node-local
export MC_SSD_SCRATCH_ROOT="${MC_SSD_SCRATCH_ROOT:-/ssd_scratch/${MC_USER}/mixed-cuts}"

# --- python environment --------------------------------------------------------
# In-repo venv (on /home2). Override MC_VENV_DIR to share one venv across checkouts.
export MC_VENV_DIR="${MC_VENV_DIR:-${MC_REPO_ROOT}/.venv}"
export MC_PYTHON_VERSION="${MC_PYTHON_VERSION:-3.12}"

# --- model + data staging (populated by `make prefetch` on the login node) ----
export MC_MODEL_NAME="${MC_MODEL_NAME:-Qwen/Qwen3-1.7B}"          # or Qwen/Qwen3-1.7B-Base
export MC_MODELS_DIR="${MC_MODELS_DIR:-${MC_STAGE_ROOT}/models}"
export MC_MODEL_DIR="${MC_MODEL_DIR:-${MC_MODELS_DIR}/$(basename "${MC_MODEL_NAME}")}"
export MC_DATA_DIR="${MC_DATA_DIR:-${MC_STAGE_ROOT}/data}"        # parquet in verl schema
export MC_RUNS_DIR="${MC_RUNS_DIR:-${MC_STAGE_ROOT}/runs/${MC_USER}}"   # durable: small outputs of every run
# Where checkpoints and the live run dir go (docs/decisions/009). /share1 has a 25 GB quota per
# user on Ada and one full-fine-tune checkpoint is ~21 GB, so the default is node-local scratch;
# the small outputs (metrics, stats, dumps, W&B) are mirrored to MC_RUNS_DIR every few minutes.
#   scratch : run dir = $MC_SCRATCH_RUNS_DIR/<run>, resubmissions must land on the same node (-w)
#   share1  : run dir = $MC_RUNS_DIR/<run> (only if the quota is ever raised to >= ~150 GB)
export MC_CHECKPOINT_HOME="${MC_CHECKPOINT_HOME:-scratch}"
export MC_SCRATCH_RUNS_DIR="${MC_SCRATCH_RUNS_DIR:-${MC_SCRATCH_ROOT}/runs}"
export MC_MIRROR_INTERVAL="${MC_MIRROR_INTERVAL:-600}"                  # seconds between mirrors

# --- caches ---------------------------------------------------------------------
# Compute nodes: node-local /scratch. Login node (no /scratch): $HOME/.cache, NOT /share1:
# /share1 allows only ~3,000 files per user and a uv-managed Python alone has thousands
# (measured 2026-09-27: "Disk quota exceeded" at 76 KB used). /home2 allows 300k files.
if [ -d "/scratch" ] && mkdir -p "${MC_SCRATCH_ROOT}" 2>/dev/null; then
  export MC_CACHE_ROOT="${MC_CACHE_ROOT:-${MC_SCRATCH_ROOT}/cache}"
  export MC_ON_LOGIN_NODE=0
else
  export MC_CACHE_ROOT="${MC_CACHE_ROOT:-${HOME}/.cache/mixed-cuts}"
  export MC_ON_LOGIN_NODE=1
  # The login node caps virtual memory at 512 MB per process (ulimit -v, hard) and 200 processes.
  # Rust/multithreaded tools die there ("memory allocation failed"): keep downloads single-path.
  export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
  export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1 RAYON_NUM_THREADS=1
fi
export HF_HOME="${HF_HOME:-${MC_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # forced to 0 on the login node above
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MC_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${MC_CACHE_ROOT}/torchinductor}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MC_CACHE_ROOT}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MC_CACHE_ROOT}/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${MC_CACHE_ROOT}/uv-python}"
export RAY_TMPDIR="${RAY_TMPDIR:-${MC_CACHE_ROOT}/ray}"
export TMPDIR="${TMPDIR:-${MC_CACHE_ROOT}/tmp}"
export WANDB_DIR="${WANDB_DIR:-${MC_CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${MC_CACHE_ROOT}/wandb-cache}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${PIP_CACHE_DIR}" \
         "${UV_CACHE_DIR}" "${RAY_TMPDIR}" "${TMPDIR}" "${WANDB_DIR}" 2>/dev/null || true

# --- logging: local jsonl is primary, W&B is offline by default --------------
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-mixed-cuts}"

# --- SLURM ------------------------------------------------------------------
export MC_SLURM_ACCOUNT="${MC_SLURM_ACCOUNT:-nlp}"
export MC_SLURM_QOS="${MC_SLURM_QOS:-normal}"
export MC_SLURM_PARTITION="${MC_SLURM_PARTITION:-u22}"
export MC_SLURM_CONSTRAINT="${MC_SLURM_CONSTRAINT:-2080ti}"
export MC_SLURM_GPUS="${MC_SLURM_GPUS:-4}"
export MC_SLURM_CPUS="${MC_SLURM_CPUS:-40}"
export MC_SLURM_MEM_PER_CPU="${MC_SLURM_MEM_PER_CPU:-3000M}"   # u22 MaxMemPerCPU=3000 MB (3G = 3072 is rejected); 40 x 3000M = 117 GB
export MC_SLURM_SIGNAL_SECS="${MC_SLURM_SIGNAL_SECS:-300}"  # SIGUSR1 this many seconds before kill

# --- runtime knobs for sm_75 / fp16 ---------------------------------------------
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" # auto-disabled on cc<8.0 anyway
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Do NOT set VLLM_USE_V2_MODEL_RUNNER: custom logits processors require Model Runner V1 and
# vLLM 0.24.0 falls back to it automatically; forcing V2 makes engine start-up raise.
unset VLLM_USE_V2_MODEL_RUNNER

# --- secrets / local overrides -------------------------------------------------
[ -f "${MC_REPO_ROOT}/.env" ] && set -a && . "${MC_REPO_ROOT}/.env" && set +a
[ -f "${MC_REPO_ROOT}/configs/local.env.sh" ] && . "${MC_REPO_ROOT}/configs/local.env.sh"

unset _mc_repo_root
