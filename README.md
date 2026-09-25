# Mixed-CUTS on verl + vLLM

Research code for stress-testing **Mixed-CUTS** from *Too Correct to Learn: Reinforcement Learning on
Saturated Reasoning Data* (Liang et al., ACL 2026, [arXiv 2604.18493](https://arxiv.org/abs/2604.18493)).
There is no public implementation; this repo implements CUTS (Constrained Uniform Top-K Sampling) as a
vLLM logits processor, schedules mixed standard/CUTS rollout groups inside verl's GRPO loop, and logs the
diagnostics that say *why* a run behaves the way it does (advantage collapse, the Eq. 5 variance
decomposition, candidate-set sizes), not just final accuracy.

Two deliverables, on **one node with 4x RTX 2080 Ti (11 GiB, sm_75, fp16 only)** under SLURM:

| Arm | Config | Group of 16 |
|---|---|---|
| vanilla GRPO | `configs/train/math_grpo.yaml` | 16 standard rollouts |
| Mixed-CUTS | `configs/train/math_mixed_cuts.yaml` | 8 standard + 8 CUTS rollouts (K=5, delta=0.03, T_warm=5) |

Both arms spend the same generation budget (asserted at config load), so any difference is the method.
Decisions the brief left open are recorded in [`docs/decisions/`](docs/decisions/README.md).

> **Status**: implemented and CPU-verified, not yet run on the cluster. Verified on a laptop: 103 CPU
> tests (`make test`), Hydra composition + invariants of every training config against verl v0.9.0's
> config tree (`make compose-check`), and the chat-template check against the real Qwen3-1.7B tokenizer.
> NOT yet verified: anything that needs a GPU: the logits processor in a live engine, the verl hooks end
> to end, fp16 stability, vLLM sleep mode on sm_75, throughput, resume. The order of first runs is in
> section 6. Items marked **TODO(cluster)** need a measurement from Ada.

---

## 1. Target platform: IIIT-H Ada, 2080 Ti nodes only

| Fact | Value | Consequence |
|---|---|---|
| GPU | 4x NVIDIA RTX 2080 Ti per node, **sm_75 (Turing), 11 GiB** each | `-C 2080ti -N 1` on every job. Never the 1080 Ti nodes (sm_61: unsupported by CUDA 13 and vLLM). |
| Driver | 580.178.04 (CUDA 13.0) | default PyPI wheels of torch 2.11.0 / vLLM 0.24.0 (CUDA 13 builds). |
| OS on compute nodes | Ubuntu 22.04.5, glibc 2.35 | `manylinux_2_28` wheels install natively; no container needed. |
| Login node | CentOS 7, glibc 2.17 | **Never** `pip install`, build, or benchmark there. Only `git`, `make prefetch`, `wandb sync`. |
| Precision | **fp16 only** (bf16 needs sm_80) | every bf16 default in verl/vLLM is overridden and asserted (section 5). |
| Attention (rollout) | vLLM **`TRITON_ATTN`**, set explicitly | `FLASH_ATTN` and `FLASHINFER` require capability 8.0 in vLLM 0.24.0; XFORMERS no longer exists ([004](docs/decisions/004-attention-backend-and-engine-version.md)). |
| Attention (training) | HF `sdpa`, `use_remove_padding: false` | FlashAttention-2 needs sm_80; `flash-attn` is not installed. |
| Engine | vLLM V1 engine (V0 is gone); **Model Runner V1** is forced by the custom logits processor | never set `VLLM_USE_V2_MODEL_RUNNER`; the preflight and the job logs check it. |
| Memory | 11 GiB/GPU, 128 GB host RAM, 40 cores | section 4; request `--mem-per-cpu=3G` (120 GB). |
| SLURM | `-A nlp --qos=normal -p u22 -C 2080ti -N 1 --gres=gpu:4 -c 40`; 4 GPUs/job, 12 across the group; interactive `srun` capped at 6 h | all real runs are `sbatch`; templates in `slurm/`. |
| Storage | `/home2/$USER` 25 GB NFS (code+venv only); `/share1` durable; `/scratch` node-local, purged after ~7 days | section 3. |

**TODO(cluster)**, detected at runtime by `scripts/check_env.py` and recorded in each job's `env_facts.json`:
whether compute nodes reach huggingface.co / api.wandb.ai; whether `/share1` is mounted on compute nodes
(the job aborts early if not); `MaxMemPerCPU` on `u22`; the attention backend and model runner vLLM
actually chose.

## 2. Pins

Version drift between verl and vLLM is the most likely thing to break this project. Every pin was
cross-checked against verl v0.9.0 (`setup.py`, `requirements.txt`, `docker/Dockerfile.stable.vllm`, CI image
`verlai/verl:vllm024.dev2`) and vLLM v0.24.0 (`requirements/cuda.txt`). Do not bump one without the others.

| Package | Pin | Why this one |
|---|---|---|
| Python | 3.12 | verl's CI image; vLLM supports 3.10-3.13 |
| CUDA runtime | 13.0 | driver 580 -> default wheels |
| **verl** | **v0.9.0** (git tag, installed `--no-deps`) | latest release; V1 trainer default; requires vLLM >= 0.18 |
| **vLLM** | **0.24.0** (default PyPI wheel = cu130) | the version in verl's CI image; wheel ships sm_75 kernels |
| torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 | exactly what vLLM 0.24.0 requires; cu130 wheel arch list includes 7.5 |
| transformers | 5.5.3 | verl `>=5.5.3,!=5.6.0,<5.11`; vLLM `>=5.5.3` |
| ray[default] | 2.56.1 | last release before verl's vllm024 image; verl needs >= 2.47 |
| tensordict | 0.10.0 | verl `>=0.8,<=0.10,!=0.9`; TransferQueue `>=0.10` |
| TransferQueue | 0.1.8 | verl hard pin |
| flashinfer-python / -cubin | 0.6.12 | vLLM hard dependency (its sampler is auto-disabled on sm_75) |
| math-verify | 0.9.0 (+ latex2sympy2_extended 1.11.0) | latest release |
| flash-attn | **not installed** | needs sm_80 |

Files: `requirements/base.txt` (hand pins, cluster), `requirements/dev.txt` (CPU tests),
`requirements/prefetch.txt` (login node downloads) and `requirements/lock.txt` (full freeze, generated on a
2080 Ti node by `make lock`; commit it after the first successful `make setup`).

## 3. Storage layout and job lifecycle

```
/home2/$USER/<repo>                       code + .venv                          (25 GB quota: NOTHING else)
/share1/$USER/mixed-cuts/                 MC_STAGE_ROOT                         durable
  models/Qwen3-1.7B/                        staged weights                      (make prefetch, login node)
  data/*.parquet + MANIFEST.json            verl-schema datasets                (make data)
  runs/$USER/<config>-s<seed>/              ONE dir per run, stable across resubmissions:
    checkpoints/global_step_N/                verl FSDP checkpoints + latest_checkpointed_iteration.txt
    metrics.jsonl  phases.jsonl  gpu_mem.jsonl  cuts_stats/  rollout_dumps/  wandb/  memory_profile.md
    jobs/<slurm job id>/                      per-submission logs, preflight report, env_facts.json
/scratch/$USER/mixed-cuts/                MC_SCRATCH_ROOT (node-local, purged): staged model/data copies + caches
```

Every `slurm/*.sbatch` does: `source configs/ada.env.sh` -> preflight (`scripts/check_env.py`, aborts
early) -> stage-in (model + parquet into `/scratch`) -> run -> on exit or on SLURM's `SIGUSR1` (sent 300 s
before a kill): stop the GPU sampler, record the engine facts, copy this job's log next to the run,
prune checkpoints. Resubmitting the same `CONFIG` + `SEED` **resumes** from the newest checkpoint
([005](docs/decisions/005-run-naming-resume-and-checkpoints.md)). All paths live in
`configs/ada.env.sh`; nothing in `src/` hardcodes `/scratch` or `/share1`.

## 4. Memory plan: one node, 4x 11 GiB

Model facts: 28 layers, hidden 2048, 16 query / 8 KV heads x 128, vocab 151,936, tied embeddings ->
**1.72 B params**; fp16 weights **3.44 GB**; KV cache **112 KB/token**. Full derivation and the
alternatives in [002](docs/decisions/002-four-gpu-memory-layout.md).

**Rollout**: one vLLM engine with tensor parallel 4 (weights 0.86 GB/card), `gpu_memory_utilization 0.40`
(~3.5 GB KV per card, ~125k tokens in total, ~20 concurrent 6k-token sequences). The engine sleeps
(level 2, weights and KV freed) during the update and is re-synced after every optimizer step.

**Training (plan A, default)**: FSDP2 sharded over the 4 ranks with `offload_policy: true`: params,
gradients and Adam state live in pinned host RAM (6.9 + 6.9 + 13.8 GB) and the optimizer step runs on the
CPU; each GPU holds one all-gathered fp16 layer plus the activations of one 6144-token micro-batch
(gradient checkpointing, chunked fp32 log-softmax): **~5-6 GB peak**. The reference model (needed by the
KL term) uses the same policy. Effective batch = 128 prompts x 16 = 2048 sequences per step, mini-batch
32 prompts (4 optimizer steps), micro-batches of <= 6144 tokens packed by `use_dynamic_bsz`.

**Plan A'** (`memory=plan_a_manual_offload`): verl's manual `param_offload`/`optimizer_offload` (update on
GPU, ~7 GB/card). **Plan B** (`memory=plan_b_lora`): LoRA r=64, documented fallback only; it weakens the
reproduction claim.

**TODO(cluster)**: `make sbatch-bench` (tokens/s and peak memory for TP=4 and TP=1) and the smoke run's
`memory_profile.md` (peak per GPU per phase, measured step time, the `save_freq` for ~30 min) decide
`gpu_memory_utilization`, `max_num_seqs` and `save_freq`. A step is plausibly 40-90 min: 2048 sequences
of up to 5000 tokens on Triton attention.

## 5. fp16 audit and stability

verl v0.9.0 defaults to bf16 in `rollout.dtype` and the actor/ref `fsdp_config.dtype`; `base_grpo.yaml`
sets `float16` everywhere with `mixed_precision {param fp16, reduce fp32, buffer fp32}`, and
`make compose-check` fails on any `bf16` string under `actor_rollout_ref`. verl's FSDP engine creates a
`ShardedGradScaler` for fp16 params. The watchlist ([006](docs/decisions/006-fp16-stability-ladder.md))
alerts every step on NaN/inf in actor metrics, gradient-norm spikes and early entropy collapse
(`stability/alert_*`), and the ladder is: LR -> 5e-7, tighter `grad_clip`, then revisit the KL term.

## 6. Quickstart on Ada

```bash
# 0. login node (CentOS 7: downloads and git only). Code lives on /home2.
git clone <this repo> ~/mixed-cuts && cd ~/mixed-cuts && cp .env.example .env   # add HF_TOKEN for GPQA
curl -LsSf https://astral.sh/uv/install.sh | sh                                    # once, if uv is missing
source configs/ada.env.sh
make setup-login && make prefetch      # pure downloads -> /share1/$USER/mixed-cuts/{models,raw}

# 1. build the environment ON A COMPUTE NODE; also builds the parquet data and runs the vLLM/verl tests
make sbatch-setup
git add requirements/lock.txt && git commit -m "lock cluster env"

# 2. measure before committing compute (each job writes to /share1/$USER/mixed-cuts/runs/$USER/...)
make sbatch-bench                      # tokens/s + peak memory: TP=4 layout and TP=1; attention backend line
make sbatch-smoke                      # 20 MATH problems, 2 steps, groups of 4 (2 std + 2 CUTS), plan A, TP=4
MC_SMOKE_RUN_DIR=/share1/$USER/mixed-cuts/runs/$USER/smoke-s42-<job> make gpu-test   # inside an allocation
make sbatch-resume-test                # kills itself at step 15, resubmits, checks it resumed at 16

# 3. the two arms (resubmit the same command to resume; use SEED=1,2,3 later for three seeds per arm)
make sbatch-train CONFIG=math_grpo SEED=1
make sbatch-train CONFIG=math_mixed_cuts SEED=1

# 4. evaluate a checkpoint (16 samples per problem; pass@1, pass@16, maj@16 with 95% CIs)
scripts/merge_ckpt.sh /share1/$USER/mixed-cuts/runs/$USER/math_mixed_cuts-s1/checkpoints   # FSDP -> HF
make sbatch-eval CKPT=/share1/$USER/mixed-cuts/runs/$USER/math_mixed_cuts-s1/checkpoints/hf/global_step_100 SEED=0
```

`make preflight` prints the full report (pins, GPU, single node, engine version, attention backend, storage,
the composed config, the chat template). `make compose-check` validates every config without a GPU.
Offline W&B: `wandb sync /share1/$USER/mixed-cuts/runs/$USER/<run>/wandb/offline-run-*` from the login
node; `metrics.jsonl` in the run dir is the primary log.

## 7. Reproduction targets (CUTS paper, Qwen3-1.7B non-thinking, trained on MATH)

Pass@1 (Table 1):

| | MATH-500 | AIME24 | AIME25 | AMC | GPQA |
|---|---|---|---|---|---|
| GRPO | 83.6 | 29.5 | 22.8 | 59.8 | 34.2 |
| Mixed-CUTS | 85.1 | 32.3 | 28.1 | 62.7 | 36.0 |
| delta | +1.5 | +2.8 | **+5.3** | +2.9 | +1.8 |

maj@16 (Table 6): GRPO 89.3 / 45.7 / 29.7 / 71.0 / 36.5; Mixed-CUTS 90.4 / 49.2 / 36.9 / 74.3 / 38.0.

Read our numbers with the benchmark sizes in mind: AIME24 and AIME25 have 30 problems (**one problem =
3.3 pp**), AMC23 has 40 (2.5 pp). The harness reports pass@1 as the mean over 16 samples with bootstrap
95% CIs over problems, never a single greedy decode, and the eval decoding follows the paper's validation
settings (T=1.0, top_p 0.8, top_k 20, 5000 tokens). Compare arms only across several seeds. Under the
boxed-plus-digit validity rule ([008](docs/decisions/008-digit-rule-and-filtered-prompts.md)) each
benchmark also reports its accuracy ceiling (fraction of ground truths that contain a digit).

## 8. What gets logged (diagnostics glossary)

Per training step (W&B + `metrics.jsonl`), in addition to verl's own `actor/entropy`, `actor/grad_norm`,
`actor/pg_loss`, `actor/kl_loss`, `response_length/mean`, `actor/perf/max_memory_allocated_gb`, timing:

| Key | Meaning |
|---|---|
| `mixed_cuts/advantage_collapse_rate` | ACR (He et al., Def. 4.1): fraction of prompt groups whose G rewards have population std < 1e-6 (advantage exactly 0) |
| `mixed_cuts/all_correct_frac`, `all_wrong_frac` | the two collapse modes separately |
| `mixed_cuts/ACR_100` | running mean of ACR over the first 100 steps (predicts final accuracy in He et al.) |
| `mixed_cuts/std_subgroup_collapse_rate`, `cuts_subgroup_collapse_rate`, `frac_groups_saved_by_cuts` | collapse of each half alone; groups where only the standard half collapsed |
| `mixed_cuts/decomp/{mu_std, mu_cuts, var_std, var_cuts, between, sigma2_mixed, identity_residual}` | Eq. 5 terms per group, averaged: `sigma2_mixed = 0.5(var_std + var_cuts) + 0.25(mu_std - mu_cuts)^2`; if `mu_cuts ~ mu_std` the between term vanishes and the mechanism does nothing |
| `mixed_cuts/rollouts_generated_per_step`, `sample_utilization`, `cost_multiplier` | He et al. Table 5 accounting; both arms read 1.0x and utilization = fraction of rollouts in non-collapsed groups |
| `reward/{mean,var,acc,group_entropy}`, `reward/{std,cuts}/{mean,var,acc,n}`, `response_length/{std,cuts}/{mean,max}` | overall and per sub-group |
| `cuts/set_size_mean`, `cuts/set_size_hist/k=i`, `cuts/degenerate_to_greedy_rate`, `cuts/frac_steps_fallback` | \|S_t\| after the delta filter (mean, histogram, fraction of steps with \|S_t\| = 1, fraction where the filter emptied the set); summarised one step late (`cuts/stats_step`) |
| `cuts/frac_old_logprob_uniform_like_k2plus` | D1 guard: fraction of CUTS tokens whose `old_log_probs` equals `-log(j)`; ~0 when the actor recomputes them, ~1 if engine logprobs leaked in (the run stops at step 1 above 0.5) |
| `stability/alert_{nan,grad_spike,entropy_collapse}` | fp16 watchlist alerts (0/1) |
| `mixed_cuts/frac_responses_with_think_tag` | must be 0 (Qwen3 non-thinking; asserted at step 1) |

A few whole groups (standard and CUTS siblings side by side) are dumped per step as JSONL under
`rollout_dumps/` for manual reading.

## 9. Repository layout

```
configs/ada.env.sh         all cluster paths / env vars         configs/train/*.yaml   Hydra overrides on verl's ppo_trainer
configs/train/memory/      plan A (default), A', B              configs/eval/*.yaml    eval decoding (paper validation settings)
src/cuts/                  CUTS operator, state, stats channel, vLLM logits processor (verl-independent, CPU-tested)
src/mixed_cuts/            verl integration: scheduler, agent loop hook, trainer, diagnostics, reward, GPQA parser,
                           stability watch, non-thinking check, entry point
src/mc_data/  src/mc_eval/ datasets -> verl parquet schema; pass@1 / pass@16 / maj@16 harness (eval/run_eval.py)
scripts/                   check_env (preflight), compose_config, prefetch, prepare_data, bench_rollout,
                           profile_memory, check_resume, merge_ckpt
slurm/                     common.sh + sbatch templates (setup, bench, smoke, train, eval, test_resume)
tests/  tests/gpu/         CPU unit tests; GPU tests run after the smoke test
docs/decisions/            numbered decision records
```

## 10. Developing on a laptop / the login node

```bash
make setup-dev      # Python 3.12 venv with torch (CPU), math-verify, pytest; no verl/vLLM
make test           # CPU unit tests (verl/vLLM-dependent ones skip)
make lint
MC_VERL_CONFIG_DIR=/path/to/verl/trainer/config make compose-check   # against a verl v0.9.0 checkout
```

## 11. How the pieces hook into verl v0.9.0 (for reviewers)

- **CUTS operator** `src/cuts/operator.py`: pure torch, batched; survivors get logit 0, everything else
  `torch.finfo(dtype).min` (never `-inf`: it turns into NaN through fp16 normalisation).
- **vLLM logits processor** `src/cuts/vllm_logits_processor.py`: V1 interface
  (`vllm.v1.sample.logits_processor.LogitsProcessor`), activated per request by
  `SamplingParams.extra_args["cuts"]`, registered through `engine_kwargs.vllm.logits_processors`. Applied
  before temperature, so `delta` acts on temperature-1 probabilities as in the paper.
- **Mixed scheduling** `src/mixed_cuts/agent_loop.py`: subclass of verl's `AgentLoopWorkerTQ` overriding
  `_run_prompt` (the method that spawns the `n` rollouts of a prompt) to emit `n_std` standard and `n_cuts`
  CUTS sessions with the same `uid`; verl's GRPO advantage groups by `uid`, so the mixed group is normalised
  together. `n_cuts: 0` reproduces the base method call for call (tested); with `n_cuts: 8` exactly 8 of 16
  requests carry the CUTS payload (tested).
- **No importance correction** ([001](docs/decisions/001-cuts-proposal-distribution.md)):
  `calculate_log_probs: false`, `rollout_correction.rollout_is: null`, `bypass_mode: false`; the actor
  recomputes `old_log_probs`, and a step-1 detector stops the run if they ever look like `log(1/|S_t|)`.
- **Diagnostics and safety** `src/mixed_cuts/trainer.py`: `PPOTrainerSync` subclass registered as
  `trainer.v1.trainer_mode: mixed_cuts_sync`; overrides `_compute_old_log_prob` (non-thinking assertion,
  D1 detector), `_compute_metrics` (D5/D6, stats, ACR_100, stability, `metrics.jsonl`), the rollout dump,
  and adds phase markers for the memory profiler.
