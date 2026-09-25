# ---------------------------------------------------------------------------
# Mixed-CUTS on verl + vLLM.  `make help` lists targets.
#
# Two environments:
#   dev      : laptop / login node, CPU only, runs the unit tests   (requirements/dev.txt)
#   cluster  : a 2080 Ti compute node, full stack                    (requirements/base.txt)
# All cluster paths come from configs/ada.env.sh (sourced below via a sub-shell).
# ---------------------------------------------------------------------------
SHELL := /bin/bash
.DEFAULT_GOAL := help

ENV_FILE   := configs/ada.env.sh
VENV       ?= $(shell . $(ENV_FILE) >/dev/null 2>&1; echo $${MC_VENV_DIR:-.venv})
PY         := $(VENV)/bin/python
UV         ?= uv
PYTHON_VER ?= 3.12
VERL_TAG   ?= v0.9.0
VERL_URL   := git+https://github.com/volcengine/verl.git@$(VERL_TAG)

# training / eval knobs (override on the command line: make train CONFIG=math_mixed_cuts)
CONFIG ?= smoke
SEED   ?= 42
CKPT   ?=
BENCH  ?= math500,aime24,aime25,amc23,gpqa_diamond
N_SAMPLES ?= 16

.PHONY: help setup-dev setup-login setup lock test gpu-test lint compose-check preflight check-env prefetch data \
        bench smoke train eval sbatch-smoke sbatch-train sbatch-eval sbatch-bench sbatch-setup sbatch-resume-test clean-cache

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

# ---------------------------------------------------------------- environments
setup-dev: ## CPU dev env (laptop / login node): venv + requirements/dev.txt + this package
	$(UV) venv --python $(PYTHON_VER) $(VENV)
	$(UV) pip install --python $(PY) -r requirements/dev.txt
	$(UV) pip install --python $(PY) --no-deps -e .
	@echo "dev env ready: $(VENV)"

setup-login: ## tiny download-only env for `make prefetch` on the CentOS 7 login node (no torch/pyarrow)
	$(UV) venv --python $(PYTHON_VER) .venv-login
	$(UV) pip install --python .venv-login/bin/python -r requirements/prefetch.txt
	@echo "login env ready: .venv-login (used automatically by make prefetch)"

setup: ## FULL cluster env. Run ONLY inside a 2080 Ti allocation (sbatch slurm/setup_env.sbatch)
	@. $(ENV_FILE); \
	  if [ -z "$$SLURM_JOB_ID" ]; then echo "ERROR: run inside a SLURM allocation on a 2080ti node, never on the login node"; exit 1; fi; \
	  $(UV) venv --python $(PYTHON_VER) $$MC_VENV_DIR && \
	  $(UV) pip install --python $$MC_VENV_DIR/bin/python -r requirements/base.txt && \
	  $(UV) pip install --python $$MC_VENV_DIR/bin/python --no-deps "verl @ $(VERL_URL)" && \
	  $(UV) pip install --python $$MC_VENV_DIR/bin/python --no-deps -e . && \
	  $$MC_VENV_DIR/bin/python scripts/check_env.py --no-staged && \
	  echo "cluster env ready: $$MC_VENV_DIR"

lock: ## freeze the resolved cluster env into requirements/lock.txt (run on the cluster after `make setup`)
	@. $(ENV_FILE); $$MC_VENV_DIR/bin/python -m pip freeze --exclude-editable > requirements/lock.txt 2>/dev/null \
	  || $(UV) pip freeze --python $$MC_VENV_DIR/bin/python > requirements/lock.txt
	@echo "wrote requirements/lock.txt ($$(wc -l < requirements/lock.txt) lines); commit it"

# ---------------------------------------------------------------------- checks
test: ## run CPU unit tests (verl/vLLM-dependent tests auto-skip)
	$(PY) -m pytest -q --ignore=tests/gpu

gpu-test: ## GPU tests (inside an allocation, after the smoke test): MC_SMOKE_RUN_DIR=... make gpu-test
	@. $(ENV_FILE); $(PY) -m pytest -q tests/gpu -m "needs_gpu" -p no:cacheprovider

lint: ## ruff
	$(PY) -m ruff check src tests scripts eval
	$(PY) -m ruff format --check src tests scripts eval

preflight: ## preflight report: pins, GPU (sm_75), single node, vLLM engine/backend, storage, config, chat template
	@. $(ENV_FILE); $(PY) scripts/check_env.py --config $(CONFIG)

check-env: preflight ## alias of preflight

compose-check: ## compose every training config (Hydra) and assert the sm_75/fp16/CUTS invariants
	@for c in base_grpo smoke math_grpo math_mixed_cuts; do \
	  $(PY) scripts/compose_config.py $$c --check $(if $(VERL_CONFIG_DIR),--verl-config-dir $(VERL_CONFIG_DIR),) || exit 1; done
	@$(PY) scripts/compose_config.py math_mixed_cuts --check $(if $(VERL_CONFIG_DIR),--verl-config-dir $(VERL_CONFIG_DIR),) memory=plan_b_lora

# ------------------------------------------------------------------------ data
prefetch: ## LOGIN NODE (has internet): download model + datasets into $$MC_STAGE_ROOT (idempotent)
	@. $(ENV_FILE); PYX=$$( [ -x .venv-login/bin/python ] && echo .venv-login/bin/python || echo $(PY) ); $$PYX scripts/prefetch.py --extra-model Qwen/Qwen3-1.7B-Base

data: ## build parquet files in verl schema (MATH, DAPO deduped, eval sets, smoke subsets)
	@. $(ENV_FILE); $(PY) scripts/prepare_data.py

# ------------------------------------------------------------------- run (node)
bench: ## rollout throughput + peak memory on ONE GPU (inside an allocation)
	@. $(ENV_FILE); $(PY) scripts/bench_rollout.py

smoke: ## end-to-end smoke test on ~20 MATH problems (inside an allocation)
	@. $(ENV_FILE); MC_SEED=$(SEED) MC_RUN_NAME=smoke-s$(SEED)-$$$$ bash slurm/run_train.sh smoke

train: ## train with configs/train/$(CONFIG).yaml (inside an allocation): make train CONFIG=math_grpo SEED=1
	@. $(ENV_FILE); MC_SEED=$(SEED) bash slurm/run_train.sh $(CONFIG)

eval: ## eval CKPT (HF dir) on BENCH with N_SAMPLES samples per problem
	@. $(ENV_FILE); $(PY) eval/run_eval.py --ckpt "$(CKPT)" --benchmarks "$(BENCH)" --n-samples $(N_SAMPLES) --seed $(SEED)

# ------------------------------------------------------------------ sbatch wrappers
sbatch-setup: ## submit the environment build job
	@. $(ENV_FILE); sbatch slurm/setup_env.sbatch
sbatch-bench: ## submit the 1-GPU rollout benchmark
	@. $(ENV_FILE); sbatch slurm/bench_rollout.sbatch
sbatch-smoke: ## submit the smoke test
	@. $(ENV_FILE); sbatch slurm/smoke.sbatch
sbatch-train: ## submit (or resume) a training run: make sbatch-train CONFIG=math_mixed_cuts SEED=1
	@. $(ENV_FILE); sbatch -J $(CONFIG)-s$(SEED) --export=ALL,MC_CONFIG=$(CONFIG),MC_SEED=$(SEED) slurm/train.sbatch
sbatch-eval: ## submit an eval run: make sbatch-eval CKPT=... BENCH=... SEED=0
	@. $(ENV_FILE); sbatch --export=ALL,MC_CKPT="$(CKPT)",MC_BENCH="$(BENCH)",MC_N_SAMPLES=$(N_SAMPLES),MC_SEED=$(SEED) slurm/eval.sbatch
sbatch-resume-test: ## kill-at-step-15-and-resubmit verification of checkpoint resume
	@. $(ENV_FILE); sbatch slurm/test_resume.sbatch

clean-cache: ## remove node-local caches (safe; they are rebuilt)
	@. $(ENV_FILE); rm -rf "$$MC_CACHE_ROOT"
