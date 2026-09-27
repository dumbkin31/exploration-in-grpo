#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Single source of truth for every cluster path and environment variable.
# Sourced by the Makefile, every slurm/*.sbatch and scripts/check_env.py (via env).
#
#   source configs/ada.env.sh
#
# Rules (see README "Storage", docs/decisions/009 and 010):
#   /home2/$USER   25 GB / 300k files NFS, on EVERY node -> code + venv, staged model/datasets,
#                                                          durable run outputs (everything but checkpoints)
#   /share1        a local disk of the LOGIN NODE          -> not mounted on compute nodes; unused
#   /scratch       node-local, purged                      -> working copies, caches, live run dir + checkpoints
# Nothing in src/ may hardcode a cluster path; it all comes from here.
# Override any variable by exporting it BEFORE sourcing this file, or by putting
# exports in configs/local.env.sh (git-ignored), which is sourced last.
# ---------------------------------------------------------------------------

_mc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$_mc_repo_root}"

# --- who / where -------------------------------------------------------------
export MC_USER="${MC_USER:-${USER:-$(id -un)}}"
export MC_STAGE_ROOT="${MC_STAGE_ROOT:-${HOME}/mixed-cuts-data}"          # durable NFS, visible on every node
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
export MC_RUNS_DIR="${MC_RUNS_DIR:-${MC_STAGE_ROOT}/runs}"              # durable: small outputs of every run
# Where checkpoints and the live run dir go (docs/decisions/009, 010). /home2 has a 25 GB quota per
# user on Ada and one full-fine-tune checkpoint is ~21 GB, so the default is node-local scratch;
# the small outputs (metrics, stats, dumps, W&B) are mirrored to MC_RUNS_DIR every few minutes.
#   scratch : run dir = $MC_SCRATCH_RUNS_DIR/<run>, resubmissions must land on the same node (-w)
#   durable : run dir = $MC_RUNS_DIR/<run> (only if a durable quota of >= ~150 GB ever appears)
export MC_CHECKPOINT_HOME="${MC_CHECKPOINT_HOME:-scratch}"
export MC_SCRATCH_RUNS_DIR="${MC_SCRATCH_RUNS_DIR:-${MC_SCRATCH_ROOT}/runs}"
export MC_MIRROR_INTERVAL="${MC_MIRROR_INTERVAL:-600}"                  # seconds between mirrors

# --- caches ---------------------------------------------------------------------
# Compute nodes: node-local /scratch. Login node (no /scratch): $HOME/.cache/mixed-cuts (small: the
# login node only downloads; /home2 allows 300k files, /share1 only ~3,000 and is unused, see 010).
if [ -d "/scratch" ] && mkdir -p "${MC_SCRATCH_ROOT}" 2>/dev/null; then
  export MC_CACHE_ROOT="${MC_CACHE_ROOT:-${MC_SCRATCH_ROOT}/cache}"
  export MC_ON_LOGIN_NODE=0
else
  export MC_CACHE_ROOT="${MC_CACHE_ROOT:-${HOME}/.cache/mixed-cuts}"
  export MC_ON_LOGIN_NODE=1
  # The login node caps virtual memory at 512 MB per process (ulimit -v, hard) and 200 processes.
  # Rust/multithreaded tools die there ("memory allocation failed"): keep downloads single-path.
  export HF_HUB_DISABLE_XET=1
  export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1 RAYON_NUM_THREADS=1
fi
export HF_HOME="${HF_HOME:-${MC_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MC_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${MC_CACHE_ROOT}/torchinductor}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MC_CACHE_ROOT}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MC_CACHE_ROOT}/uv}"
# uv-managed interpreters must live on shared storage: the venv on /home2 symlinks to them, and a
# node-local /scratch copy would vanish (purge) or be missing on the next node. ~150 MB in $HOME.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"
export PATH="${HOME}/.local/bin:${PATH}"   # uv installs there; batch jobs do not source ~/.bashrc
export RAY_TMPDIR="${RAY_TMPDIR:-${MC_CACHE_ROOT}/ray}"
export TMPDIR="${TMPDIR:-${MC_CACHE_ROOT}/tmp}"
export WANDB_DIR="${WANDB_DIR:-${MC_CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${MC_CACHE_ROOT}/wandb-cache}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${PIP_CACHE_DIR}" \
         "${UV_CACHE_DIR}" "${RAY_TMPDIR}" "${TMPDIR}" "${WANDB_DIR}" 2>/dev/null || true

# --- package index -----------------------------------------------------------------
# From Ada, PyPI's CDN (files.pythonhosted.org, Fastly) is throttled to ~60-100 KB/s, and so are
# GitHub release downloads; a 10 GB torch/vLLM install would take days. Measured 2026-09-27 from a
# compute node: Tsinghua TUNA 4.5 MB/s, SJTU 2.5 MB/s, download.pytorch.org 6.7 MB/s,
# wheels.vllm.ai 7.6 MB/s, huggingface.co 3 MB/s. TUNA mirrors all of PyPI, so the pins resolve
# identically. Override with UV_DEFAULT_INDEX in configs/local.env.sh if it is ever slow.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${UV_DEFAULT_INDEX}}"

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
# Node selection: the 2080 Ti nodes run MIXED NVIDIA driver generations (docs/decisions/011). The pinned
# wheels are CUDA 13.0 builds (vLLM 0.24.0 ships only cu130 and cu129 wheels) and need driver >= 580.
# Census 2026-09-27 via /proc/driver/nvidia/version (25 of 44 nodes; the rest were busy):
#   580.178 / 595.91 (OK): gnode065 068 070 078 081 084 087
#   570.211 (CUDA 12.8):   gnode043 050 054 056 072 073 079 080 082 085 090 091
#   no driver loaded:      gnode066 076 088 089
# The driver does not follow any SLURM feature (phase3 has both), so `make sbatch-*` passes -x with this
# list plus MC_BAD_NODES_FILE, which mc_job_init appends to when a job lands on an unlisted old node.
export MC_MIN_DRIVER_MAJOR="${MC_MIN_DRIVER_MAJOR:-580}"
export MC_SLURM_EXCLUDE="${MC_SLURM_EXCLUDE:-gnode043,gnode050,gnode054,gnode056,gnode066,gnode072,gnode073,gnode076,gnode079,gnode080,gnode082,gnode085,gnode088,gnode089,gnode090,gnode091}"
export MC_BAD_NODES_FILE="${MC_BAD_NODES_FILE:-${MC_STAGE_ROOT}/bad_nodes.txt}"
mc_sbatch_exclude() {  # prints "-x <nodes>" for sbatch: the static list + every node recorded in MC_BAD_NODES_FILE
  local list="${MC_SLURM_EXCLUDE:-}" extra=""
  if [ -f "${MC_BAD_NODES_FILE:-/nonexistent}" ]; then
    extra="$(awk 'NF{print $1}' "${MC_BAD_NODES_FILE}" | sort -u | paste -s -d , -)"
  fi
  [ -n "${extra}" ] && list="${list:+${list},}${extra}"
  [ -n "${list}" ] && printf -- '-x %s' "${list}"
  return 0
}

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
