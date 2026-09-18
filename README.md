# Mixed-CUTS on verl + vLLM

Research code for stress-testing **Mixed-CUTS** from *Too Correct to Learn: Reinforcement Learning on
Saturated Reasoning Data* (Liang et al., ACL 2026, [arXiv 2604.18493](https://arxiv.org/abs/2604.18493)).
There is no public implementation; this repo implements CUTS (Constrained Uniform Top-K Sampling) as a
vLLM logits processor, schedules mixed standard/CUTS rollout groups inside verl's GRPO loop, and logs the
diagnostics that say *why* a run behaves the way it does (advantage collapse, per-subgroup rewards,
candidate-set sizes), not just final accuracy.

Phase-1 hypothesis: CUTS's benefit does not generalise beyond the saturated-data regime. So every
config runs identically with CUTS on (`mixed_cuts.n_cuts > 0`) and off (`n_cuts: 0`, vanilla GRPO).

> **Status**: scaffold, not yet run on the cluster. Verified so far, on a laptop without a GPU:
> the CPU test suite (`make test`, 70 tests: CUTS operator, per-request state machine, mixed-group
> scheduler, diagnostics, reward, data loaders, eval scoring/runner) and Hydra composition of every
> training config against verl v0.9.0's config tree (`make compose-check`). NOT yet verified: anything
> that needs a GPU or the cluster: the vLLM logits processor inside a live engine, the verl
> subclass hooks end to end, fp16 stability, sleep mode on sm_75, throughput. The smoke test
> (`make sbatch-smoke`) is the first thing that must pass. Sections marked **TODO(cluster)** need a
> measurement from Ada before they are final.

---

## 1. Target platform: IIIT-H Ada (SLURM), 2080 Ti nodes only

| Fact | Value | Consequence |
|---|---|---|
| GPU | 4x NVIDIA RTX 2080 Ti per node, **sm_75 (Turing), 11 GiB** each | `-C 2080ti` on every job. Never the 1080 Ti nodes (sm_61: unsupported by CUDA 13 and vLLM). |
| Driver | 580.178.04 (CUDA 13.0) | default PyPI wheels of torch 2.11.0 / vLLM 0.24.0 (CUDA 13 builds). |
| OS on compute nodes | Ubuntu 22.04.5, glibc 2.35 | `manylinux_2_28` wheels install natively; **no container needed**. |
| Login node | CentOS 7, glibc 2.17 | **Never** `pip install`, build, or benchmark there. Only `git`, `make prefetch`, `wandb sync`. |
| Precision | **fp16 only** (bf16 needs sm_80) | every bf16 default in verl/vLLM is overridden explicitly (section 5). |
| Attention (rollout) | vLLM picks **`TRITON_ATTN`** on sm_75 | `FLASH_ATTN` and `FLASHINFER` both require compute capability 8.0 in vLLM 0.24.0 (`vllm/v1/attention/backends/{flash_attn,flashinfer}.py::supports_compute_capability`); the Triton backend accepts any capability. FlexAttention is the next fallback. |
| Attention (training) | HF `sdpa`, `use_remove_padding: false` | FlashAttention-2 needs sm_80, so `flash-attn` is **not installed** and verl's remove-padding path (which needs FA varlen kernels) is off. |
| Sampler | vLLM's FlashInfer sampler auto-disables on cc < 8.0 | torch top-k/top-p fallback; harmless. |
| Custom logits processors | force vLLM **Model Runner V1** | vLLM 0.24.0's Model Runner V2 does not support them and falls back automatically. Do **not** set `VLLM_USE_V2_MODEL_RUNNER`. |
| Memory | 11 GiB/GPU, 128 GB host RAM, 40 cores | see section 4. Request `--mem-per-cpu=3G` (120 GB), not 2G. |
| SLURM | `-A nlp --qos=normal -p u22 -C 2080ti --gres=gpu:4 -c 40`; 4 GPUs/job, 12 GPUs across the group; interactive `srun` capped at 6 h | all real runs are `sbatch`; templates in `slurm/`. |
| Storage | `/home2/$USER` 25 GB NFS (code+venv only); `/share1` durable; `/scratch` node-local, purged after ~7 days | section 3. |

**TODO(cluster)** still unknown, detected at runtime by `scripts/check_env.py` and recorded in the job log:
whether compute nodes reach huggingface.co / api.wandb.ai (or via a proxy); whether `/share1` is mounted
on compute nodes; `MaxMemPerCPU` on `u22`; `uname -r`; whether Apptainer/Singularity exists (optional
hardening later, not a prerequisite).

## 2. Pins

Version drift between verl and vLLM is the most likely thing to break this project. Every pin below was
cross-checked against verl v0.9.0 (`setup.py`, `requirements.txt`, `docker/Dockerfile.stable.vllm`, CI image
`verlai/verl:vllm024.dev2`) and vLLM v0.24.0 (`requirements/cuda.txt`). Do not bump one without the others.

| Package | Pin | Why this one |
|---|---|---|
| Python | 3.12 | verl's CI image; vLLM supports 3.10-3.13 |
| CUDA runtime | 13.0 | driver 580 -> default wheels |
| **verl** | **v0.9.0** (git tag, installed `--no-deps`) | latest release (2026-08-14); V1 trainer is default; requires vLLM >= 0.18 |
| **vLLM** | **0.24.0** (default PyPI wheel = cu130) | the version in verl's CI image and stable Dockerfile; wheel ships sm_75 kernels (`CUDA_ARCH_X86 = 7.5 8.0 8.6 8.9 9.0 10.0 12.0`) |
| torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 | exactly what vLLM 0.24.0 requires; cu130 wheel arch list `7.5;8.0;8.6;9.0;10.0;12.0` |
| transformers | 5.5.3 | verl `>=5.5.3,!=5.6.0,<5.11`; vLLM `>=5.5.3` |
| ray[default] | 2.56.1 | last release before verl's vllm024 image (2026-07-21); verl needs >= 2.47 |
| tensordict | 0.10.0 | verl `>=0.8,<=0.10,!=0.9`; TransferQueue `>=0.10` |
| TransferQueue | 0.1.8 | verl hard pin |
| flashinfer-python / -cubin | 0.6.12 | vLLM hard dependency (its sampler is auto-disabled on sm_75) |
| math-verify | 0.9.0 (+ latex2sympy2_extended 1.11.0) | latest release |
| flash-attn | **not installed** | needs sm_80 |

Files: `requirements/base.txt` (hand pins, cluster), `requirements/dev.txt` (CPU tests), and
`requirements/lock.txt` (full freeze, generated on a 2080 Ti node by `make lock`; commit it after the
first successful `make setup`).

## 3. Storage layout and job lifecycle

```
/home2/$USER/<repo>          code + .venv              (25 GB quota: NOTHING else goes here)
/share1/$USER/mixed-cuts/    MC_STAGE_ROOT             durable
  models/Qwen3-1.7B/           staged weights (make prefetch, login node)
  data/*.parquet + MANIFEST    verl-schema datasets   (make data)
  runs/$USER/<jobid>/          checkpoints, metrics.jsonl, rollout dumps, offline W&B  (copied out by the job)
  cache/                       HF cache used by the login node
/scratch/$USER/mixed-cuts/   MC_SCRATCH_ROOT           node-local, purged
  cache/{huggingface,triton,torchinductor,pip,uv,ray,wandb}
  runs/<jobid>/                live run dir
```

Every `slurm/*.sbatch` does, in order: `source configs/ada.env.sh` -> preflight (`scripts/check_env.py`,
aborts early) -> stage-in (model + parquet from `MC_STAGE_ROOT` into `/scratch`) -> run -> stage-out
(`rsync` checkpoints/logs/W&B back to `MC_STAGE_ROOT/runs/$USER/$SLURM_JOB_ID`). Stage-out is also bound to
`SIGUSR1`/`SIGTERM`/`EXIT` and the job requests `--signal=B:SIGUSR1@300`, so a killed or timed-out run still
flushes. Outputs are namespaced by `$USER/$SLURM_JOB_ID` so four people never overwrite each other.

All paths and knobs live in `configs/ada.env.sh`; nothing in `src/` hardcodes `/scratch` or `/share1`.

## 4. Memory plan for Qwen3-1.7B on 4x 11 GiB

Model facts: 28 layers, hidden 2048, 16 query / 8 KV heads x 128, vocab 151,936, tied embeddings ->
**1.72 B params**; fp16 weights **3.44 GB**; KV cache **112 KB/token** (2 x 28 x 8 x 128 x 2 B).

**Plan A (default): full fine-tune, FSDP2 + CPU offload, fp16 mixed precision** (`configs/train/memory/plan_a_fullft_offload.yaml`)
- Host RAM (pinned): fp32 master 6.9 GB + fp32 grads 6.9 GB + Adam m/v 13.8 GB = 27.6 GB actor; ref policy 6.9 GB; Ray object store capped at 16 GB; 4 vLLM processes ~3 GB each; sleeping replicas ~0 (level-2 sleep discards weights, they are re-synced every step). Total ~70 GB of the 120 GB requested.
- GPU during training: one all-gathered fp16 layer (< 0.2 GB) + activations with gradient checkpointing, micro-batch 1, seq <= 1024 + 3072 (~2-3 GB) + fp16 logits 4096 x 151,936 (1.25 GB) + chunked fp32 log-softmax/entropy (~0.6 GB) + CUDA context -> **~5-6 GB peak**.
- GPU during rollout: 4 vLLM replicas (TP=1), weights 3.44 GB + `gpu_memory_utilization 0.70` -> ~4.2 GB KV ~ 37k tokens ~ 9 concurrent 4k sequences per card.
- Cost: CPU AdamW over 1.72 B params ~5-15 s/step; 3.44 GB weight sync per replica per step. Expected to be dwarfed by generation time.

**Plan B (switch): LoRA r=64 on all linear layers** (`plan_b_lora.yaml`, select with `memory=plan_b_lora`): sharded frozen base 0.86 GB/GPU + adapter state < 0.6 GB + activations -> ~4-5 GB; ref = adapter disabled. Faster, but a deviation from the paper (full fine-tune); if adopted it is written up as a limitation. **TODO(cluster)**: the merged-weight sync to vLLM (`model.lora.merge: true`) is unverified.

**Plan C (fallback only)**: 2 GPUs train / 2 GPUs rollout with verl's `separate_async` mode, if vLLM sleep mode misbehaves on sm_75 in the smoke test.

**Throughput is the real constraint** (no FlashAttention, < 10 concurrent sequences per card). Run `make sbatch-bench`
first: it reports tokens/s at several concurrencies and peak memory on one GPU. **TODO(cluster)**: fill in the
numbers here and choose `max_response_length` / group size `G` with them. Defaults: Qwen3 **non-thinking** mode
(`enable_thinking: false`, ~5 tokens of empty `<think>` block per response), `max_prompt_length 1024`,
`max_response_length 3072` (paper: 5000). Smoke test: 512 / 512.

## 5. fp16 audit (every bf16 default we override)

verl v0.9.0 defaults to bf16 in `rollout.dtype`, `actor.fsdp_config.dtype`, `ref.fsdp_config.dtype`,
`critic.*` (unused) and `reward.model.dtype` (unused). `configs/train/base_grpo.yaml` sets
`rollout.dtype: float16`, `{actor,ref}.fsdp_config.dtype: float16` and
`mixed_precision: {param_dtype: fp16, reduce_dtype: fp32, buffer_dtype: fp32}`. verl's FSDP engine creates a
`ShardedGradScaler` automatically for fp16 params (`verl/workers/engine/fsdp/transformer_impl.py`); the loss,
log-probs and entropy are computed in fp32. Overflow risks we watch from step 1: `actor/grad_scale` collapsing,
NaN/inf in `actor/pg_loss`, and inf logits from vLLM (checked in the rollout dump).

## 6. Quickstart on Ada

```bash
# 0. login node: code lives on /home2. The login node is CentOS 7: it only downloads and syncs.
git clone <this repo> ~/mixed-cuts && cd ~/mixed-cuts && cp .env.example .env   # add HF_TOKEN for GPQA
curl -LsSf https://astral.sh/uv/install.sh | sh                                    # once, if uv is missing
source configs/ada.env.sh
make setup-login && make prefetch      # pure downloads -> /share1/$USER/mixed-cuts/{models,raw}

# 1. build the environment ON A COMPUTE NODE (never the login node); also builds the parquet data
make sbatch-setup                      # slurm/setup_env.sbatch: venv + pins + verl@v0.9.0 + tests + `make data`
git add requirements/lock.txt && git commit -m "lock cluster env"

# 2. measure before committing compute
make sbatch-bench                      # tokens/s + peak memory on 1 GPU  -> runs/<user>/bench-<job>/bench.json
make sbatch-smoke                      # 20 MATH problems, 2 steps, n=4 (2 std + 2 CUTS), plan A

# 3. real runs (each config also works with `make train CONFIG=...` inside an interactive allocation)
make sbatch-train CONFIG=math_grpo          # vanilla GRPO (n_cuts=0)
make sbatch-train CONFIG=math_mixed_cuts    # 8 std + 8 CUTS
make sbatch-train CONFIG=dapo_mixed_cuts
scripts/merge_ckpt.sh /share1/$USER/mixed-cuts/runs/$USER/math_mixed_cuts-<job>/checkpoints   # FSDP -> HF
make sbatch-eval CKPT=/share1/$USER/mixed-cuts/runs/$USER/math_mixed_cuts-<job>/checkpoints/hf/global_step_100
```

Validate configs without a GPU: `make compose-check` (composes every `configs/train/*.yaml` against verl's
config tree and asserts no bf16, `n_std + n_cuts == rollout.n`, the CUTS logits processor and mixed agent
loop are wired, `sdpa` + no remove-padding, and the `mixed_cuts_sync` trainer).

Offline W&B: runs log with `WANDB_MODE=offline` into the run dir; after stage-out run
`wandb sync /share1/$USER/mixed-cuts/runs/$USER/<name>-<job>/wandb/offline-run-*` from the login node. The
primary log is always `metrics.jsonl` in the run dir (one JSON object per training step).

## 7. What gets logged (diagnostics glossary)

Per training step (W&B + `metrics.jsonl`), in addition to verl's own metrics (`actor/entropy`, `actor/pg_loss`,
`critic/score/*`, `response_length/*`, timing):

| Key | Meaning |
|---|---|
| `mixed_cuts/advantage_collapse_rate` | fraction of prompt groups whose rewards are all identical (normalised advantage = 0 for the whole group) |
| `mixed_cuts/frac_groups_saved_by_cuts` | groups where the standard sub-group alone collapsed but the mixed group did not |
| `reward/mean`, `reward/var`, `reward/group_entropy` | over all rollouts; entropy of the per-group reward histogram, averaged over groups |
| `reward/std/{mean,var,acc}`, `reward/cuts/{mean,var,acc}` | the same statistics restricted to standard vs CUTS rollouts |
| `response_length/std/mean`, `response_length/cuts/mean` | mean trajectory length in tokens per sub-group |
| `cuts/set_size_mean` | mean candidate-set size \|S_t\| after the filter, over all CUTS decoding steps this step |
| `cuts/frac_steps_singleton` | fraction of CUTS steps with \|S_t\| = 1 (CUTS degenerating into greedy) |
| `cuts/frac_steps_fallback` | fraction of CUTS steps where the filter emptied the set and the fallback rule fired |

A handful of whole groups (standard and CUTS siblings side by side) are dumped per step as JSONL under
`rollout_dumps/` in the run dir for manual reading.

## 8. Repository layout

```
configs/ada.env.sh        all cluster paths / env vars          configs/train/*.yaml   Hydra overrides on verl's ppo_trainer
src/cuts/                 CUTS operator + vLLM logits processor (verl-independent, CPU-tested)
src/mixed_cuts/           verl integration: mixed scheduler, diagnostics, trainer subclass, reward, entry point
src/mc_data/              dataset loaders -> verl parquet schema (MATH, DAPO-17k deduped, 5 eval sets)
src/mc_eval/  eval/       pass@1 / pass@16 / maj@16 harness on vLLM offline
scripts/                  check_env, prefetch, prepare_data, bench_rollout, merge_ckpt
slurm/                    sbatch templates + common.sh (preflight, staging, signal traps)
tests/                    CPU unit tests (vLLM/verl-dependent ones auto-skip)
```

## 9. Developing on a laptop / the login node

```bash
make setup-dev      # Python 3.12 venv with torch (CPU), math-verify, pytest; no verl/vLLM
make test           # unit tests for the operator, state machine, scheduler, diagnostics, reward, data
make lint
```

## 10. How the pieces hook into verl v0.9.0 (for reviewers)

- **CUTS operator** `src/cuts/operator.py`: pure torch, batched; inactive rows untouched.
- **vLLM logits processor** `src/cuts/vllm_logits_processor.py`: implements the V1 interface
  (`vllm.v1.sample.logits_processor.LogitsProcessor`: `update_state(BatchUpdate)` / `apply(logits)`), activated per
  request by `SamplingParams.extra_args["cuts"]`, registered declaratively via
  `actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors`. vLLM applies it before temperature, so
  `delta` acts on temperature-1 probabilities exactly as in the paper.
- **Mixed scheduling** `src/mixed_cuts/agent_loop.py`: subclass of verl's `AgentLoopWorkerTQ` overriding
  `_run_prompt` (the method that spawns the `n` rollouts of one prompt) to emit `n_std` standard and `n_cuts` CUTS
  sessions with the same prompt `uid`; plugged in through `rollout.agent.agent_loop_manager_class`. GRPO advantage
  in verl groups by `uid`, so the mixed group is normalised together without touching verl's advantage code.
  `n_cuts: 0` reproduces the base method exactly (unit-tested).
- **Diagnostics** `src/mixed_cuts/trainer.py`: `PPOTrainerSync` subclass registered as
  `trainer.v1.trainer_mode: mixed_cuts_sync`, overriding `_compute_metrics` and the rollout dump.
