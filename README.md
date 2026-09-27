# Mixed-CUTS on verl + vLLM

Research code for stress-testing **Mixed-CUTS** from *Too Correct to Learn: Reinforcement Learning on
Saturated Reasoning Data* (Liang et al., ACL 2026, [arXiv 2604.18493](https://arxiv.org/abs/2604.18493)).
There is no public implementation; this repo implements CUTS (Constrained Uniform Top-K Sampling) as a
vLLM logits processor, schedules mixed standard/CUTS rollout groups inside verl's GRPO loop, and logs the
diagnostics that say *why* a run behaves the way it does (advantage collapse, the Eq. 5 variance
decomposition, candidate-set sizes), not just final accuracy.

Two deliverables, each trained on **one RTX 2080 Ti (11 GiB, sm_75, fp16 only) with LoRA** under SLURM
(account `research`: one GPU per user, [012](docs/decisions/012-research-qos-one-gpu-lora.md)); the original
4-GPU full-fine-tune layout stays selectable (`MC_LAYOUT=nlp_4gpu`):

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
| GPU | NVIDIA RTX 2080 Ti, **sm_75 (Turing), 11 GiB**; 4 per node, **1 per job** on `research` | `-C 2080ti -N 1 --gres=gpu:1` on every job. Never the 1080 Ti nodes (sm_61: unsupported by CUDA 13 and vLLM). |
| Driver | 580.178.04 (CUDA 13.0) | default PyPI wheels of torch 2.11.0 / vLLM 0.24.0 (CUDA 13 builds). |
| OS on compute nodes | Ubuntu 22.04.5, glibc 2.35, internet works, `/scratch` 1.8 TB | `manylinux_2_28` wheels install natively; no container needed. PyPI's CDN is throttled to ~0.1 MB/s from Ada, so `configs/ada.env.sh` uses the Tsinghua mirror (4.5 MB/s). |
| Login node | CentOS 7, glibc 2.17, **512 MB virtual memory per process, 200 processes** (hard limits) | Only `git`, `make setup-login`, `make prefetch` (plain pip, no `hf_transfer`/`hf-xet`). `wandb` and `torch` wheels need glibc >= 2.28, so W&B syncing happens elsewhere (section 6). |
| Precision | **fp16 only** (bf16 needs sm_80) | every bf16 default in verl/vLLM is overridden and asserted (section 5). |
| Attention (rollout) | vLLM **`TRITON_ATTN`**, set explicitly | `FLASH_ATTN` and `FLASHINFER` require capability 8.0 in vLLM 0.24.0; XFORMERS no longer exists ([004](docs/decisions/004-attention-backend-and-engine-version.md)). |
| Attention (training) | HF `sdpa`, `use_remove_padding: false` | FlashAttention-2 needs sm_80; `flash-attn` is not installed. |
| Engine | vLLM V1 engine (V0 is gone); **Model Runner V1** is forced by the custom logits processor | never set `VLLM_USE_V2_MODEL_RUNNER`; the preflight and the job logs check it. |
| Memory | 11 GiB on the card; **30 GB host RAM and 10 cores per job** (`research`/`low` caps: `-c 10 --mem-per-cpu=3000M`) | section 4: LoRA + the FSDP2 offload policy; full fine-tuning needs the 4-card layout (117 GB). |
| SLURM | account `research` / QoS `low`: **per user 1 GPU, 10 CPUs, 32 GB, 1 node per job, 5 jobs, 4-day MaxWall**; `-A research --qos=low -p u22 -C 2080ti -N 1 --gres=gpu:1 -c 10 --mem-per-cpu=3000M` (`MaxMemPerCPU=3000`; `3G` is rejected); `nlp`/`normal` = 4 GPUs/job, 12 across the group; interactive `srun` capped at 6 h | all real runs are `make sbatch-*` (adds the layout flags + node exclude list); templates in `slurm/`; `MC_LAYOUT=nlp_4gpu` flips everything to the 4-GPU layout ([012](docs/decisions/012-research-qos-one-gpu-lora.md)). |
| NVIDIA driver | **mixed across the 2080 Ti nodes**: 580/595 on 7 of the 25 probed, 570 (CUDA 12.8) on 12, no module on 4 | the cu130 wheels need >= 580 (vLLM 0.24.0 has no cu128 wheel): `make sbatch-*` passes `-x $MC_SLURM_EXCLUDE`, and every job checks `/proc/driver/nvidia/version` before touching the run dir ([011](docs/decisions/011-mixed-driver-generations.md)). |
| Storage | `/home2/$USER` 25 GB / 300k files NFS, the **only durable file system compute nodes see**; `/share1` is a local disk of the login node (measured: absent on the gnodes); `/scratch` node-local 1.8 TB, purged after ~7 days | section 3: code, venv, staged model/data and each run's small outputs on `/home2`; checkpoints on scratch ([009](docs/decisions/009-scratch-checkpoints-quota.md), [010](docs/decisions/010-share1-is-login-node-local.md)). |

Measured on 2026-09-27 (setup job 2719025 on gnode084, a CPU diagnostic on gnode043) and re-checked by
`scripts/check_env.py` in every job (`env_facts.json`): compute nodes reach huggingface.co and api.wandb.ai;
`/share1` does not exist on compute nodes (it is a local ext4 disk of the login node); `u22` has
`MaxMemPerCPU=3000`; vLLM 0.24.0 picks `TRITON_ATTN` and Model Runner V1 on the 2080 Ti.

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
2080 Ti node by the setup job, `make sbatch-setup`; first committed 2026-09-27 from gnode084).

## 3. Storage layout and job lifecycle

```
/home2/$USER/                             NFS, 25 GB / 300k files per user, mounted on EVERY node
  mixed-cuts/ (this repo)                   code + .venv (~9.6 GB)
  mixed-cuts-data/                          MC_STAGE_ROOT (durable)
    models/Qwen3-1.7B/                        staged weights (~4.8 GB)                     (make prefetch, login node)
    raw/                                      Hub snapshots of the datasets (~0.3 GB)      (make prefetch)
    data/*.parquet + MANIFEST.json            verl-schema datasets                         (make sbatch-data)
    runs/<config>-s<seed>/                    MIRROR of each run (everything but checkpoints), every 10 min:
      metrics.jsonl  phases.jsonl  gpu_mem.jsonl  wandb/  memory_profile.md
      cuts_stats.tar.gz  rollout_dumps.tar.gz     per-step files packed (~5x smaller; the quota is space)
      jobs/<slurm job id>/                      per-submission logs, preflight report, env_facts.json
      node.txt                                  which node holds the checkpoints (resubmissions pin it)
/scratch/$USER/mixed-cuts/                MC_SCRATCH_ROOT, node-local 1.8 TB, purged after ~7 days
  runs/<config>-s<seed>/                    the LIVE run dir: checkpoints/global_step_N/ + the files above
  stage/, cache/                            staged model/data copies (re-rsynced per job), caches
/share1/$USER/                            a local disk of the LOGIN NODE: not mounted on compute nodes, unused
```

Why this split: one FSDP checkpoint is ~21 GB and `/home2`, the only durable file system compute nodes can
see ([010](docs/decisions/010-share1-is-login-node-local.md)), allows 25 GB per user, so checkpoints stay
on the node that wrote them ([009](docs/decisions/009-scratch-checkpoints-quota.md)). Consequences:
resubmitting a run must land on the same node (`make sbatch-train` reads `node.txt` and adds `-w`); LoRA
checkpoints are ~7.8 GB (full PEFT state dict), still too big for the home quota; if
that node is lost, the curves and diagnostics survive in the mirror but the run restarts from step 0.

Every `slurm/*.sbatch` does: `source configs/ada.env.sh` -> preflight (`scripts/check_env.py`, aborts
early) -> stage-in (model + parquet into `/scratch`) -> run, mirroring to `/home2` every 10 min -> on exit
or on SLURM's `SIGUSR1` (sent 300 s before a kill): stop the GPU sampler, record the engine facts, copy this
job's log next to the run, prune checkpoints, final mirror. Resubmitting the same `CONFIG` + `SEED` on the
same node **resumes** from the newest checkpoint ([005](docs/decisions/005-run-naming-resume-and-checkpoints.md)).
All paths live in `configs/ada.env.sh`; nothing in `src/` hardcodes `/scratch` or `/home2`.

## 4. Memory plan: one 2080 Ti, LoRA, 30 GB of host RAM

Model facts: 28 layers, hidden 2048, 16 query / 8 KV heads x 128, vocab 151,936, tied embeddings ->
**1.72 B params**; fp16 weights **3.44 GB**; KV cache **112 KB/token**. Why one GPU and LoRA, with the verl
source lines behind every claim: [012](docs/decisions/012-research-qos-one-gpu-lora.md). The 4-GPU full
fine-tune of [002](docs/decisions/002-four-gpu-memory-layout.md) stays selectable with `MC_LAYOUT=nlp_4gpu`.

**Training (plan B, default, `memory=plan_b_lora`)**: LoRA r=64 / alpha=128 on every linear layer (69.7 M
trainable parameters) under the FSDP2 `CPUOffloadPolicy`: the fp32 base and the LoRA masters live in pinned
host RAM (~7.2 GB), one decoder layer is gathered in fp16 at a time, gradients and Adam state exist only for
the adapter (~0.85 GB). The KL reference is the same model with the adapter disabled (no second copy).
Effective batch = 128 prompts x 16 = 2048 sequences per step, mini-batch 32 prompts (4 optimizer steps),
micro-batches of <= 6144 tokens packed by `use_dynamic_bsz`. GPU peak during the update **~9-9.5 GiB**
(vLLM asleep at level 2: weights and KV freed; fp16 logits + chunked fp32 log-softmax dominate).

**Rollout**: one vLLM engine, TP=1, `gpu_memory_utilization 0.80` while the actor sits in host RAM:
3.44 GB weights + ~1 GB activations/cudagraphs + ~4.3 GB KV (~39k tokens: 6 max-length or ~25 typical
sequences). Merged full weights are streamed to the engine after every optimizer step (`lora.merge: true`).

**Host RAM** (30 GB cgroup): FSDP worker ~10.5 GB, vLLM ~4.5, driver ~2.5, agent/TransferQueue/reward/
dataloader workers ~5, Ray ~1.5, object store 1-2 -> ~26-27 GB. Ray gets `num_cpus: 16` *logical* CPUs
because it reserves ~10 for its own actors (fewer deadlocks placement); the cgroup's 10 real cores limit use.

**Fallback ladder** if the smoke run's `memory_profile.md` / `host_mem_peak.txt` disagree: lower
`max_num_seqs` -> lower `gpu_memory_utilization` -> `actor.use_dynamic_bsz: false` with
`ppo_micro_batch_size_per_gpu: 1` -> `max_response_length` (deviates from the paper).

**TODO(cluster)**: `make sbatch-bench` (tokens/s at concurrency 1-64, KV size at util 0.80/0.85, preemptions
at 4096-token responses) and the smoke run decide `gpu_memory_utilization`, `max_num_seqs` and `save_freq`.
A step is plausibly 1-1.5 h: 2048 sequences of up to 5000 tokens on one card.

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
make setup-login && make prefetch      # Qwen3-1.7B + datasets -> ~/mixed-cuts-data/{models,raw} (~5 GB on /home2)

# 1. build the environment ON A COMPUTE NODE; also builds the parquet data and runs the vLLM/verl tests.
#    Always submit through make: it excludes the nodes whose NVIDIA driver is too old for the cu130 wheels (011).
make sbatch-setup
git add requirements/lock.txt && git commit -m "lock cluster env"
make sbatch-data                       # only if prefetch finished after the setup job ran (it builds the parquet + a full preflight)

# 2. measure before committing compute (each job mirrors to ~/mixed-cuts-data/runs/<run>/). Every
#    make sbatch-* target adds the layout's flags (-A research --qos=low --gres=gpu:1 -c 10 ...) + the exclude list.
make sbatch-bench                      # ~40 min: tokens/s at concurrency 1-64, KV size at util 0.80/0.85, TRITON_ATTN line
make sbatch-smoke                      # ~45 min: 20 MATH problems, 2 steps, groups of 4 (2 std + 2 CUTS), then tests/gpu in the same job
make sbatch-resume-test                # 2 x ~2 h: kills itself at step 15, resubmits on the same node, checks it resumed at 16

# 3. the two arms: ONE GPU PER USER on research, so one arm at a time per account (~4-6 days each, two
#    submissions under the 4-day MaxWall; resubmit the same command to resume). SEED=1,2,3 later for three seeds.
make sbatch-train CONFIG=math_grpo SEED=1
make sbatch-train CONFIG=math_mixed_cuts SEED=1

# 4. evaluate a checkpoint (16 samples per problem; pass@1, pass@16, maj@16 with 95% CIs).
#    Checkpoints live on the node named in the run's node.txt; merge (LoRA folded in) and evaluate there.
#    The merge srun counts against the user's 10 CPUs: run it while no training job of yours is running.
NODE=$(sed -n 's/^node=//p' ~/mixed-cuts-data/runs/math_mixed_cuts-s1/node.txt | tail -1)
srun -A research --qos=low -p u22 -w $NODE -c 6 --mem-per-cpu=3000M -t 01:00:00 scripts/merge_ckpt.sh /scratch/$USER/mixed-cuts/runs/math_mixed_cuts-s1/checkpoints
make sbatch-eval CKPT=/scratch/$USER/mixed-cuts/runs/math_mixed_cuts-s1/checkpoints/hf/global_step_100 NODE=$NODE SEED=0
```

`make preflight` prints the full report (pins, GPU, single node, engine version, attention backend, storage,
the composed config, the chat template). `make compose-check` validates every config without a GPU.
**W&B.** One team project (`anlp-mixed-cuts/mixed-cuts`); `WANDB_API_KEY` and `WANDB_ENTITY` live in `.env` on
Ada. Jobs log **online from the compute nodes** (they reach api.wandb.ai; measured): the run id is the run
name, so a resumed run keeps updating the same W&B run, and `metrics.jsonl` in the run dir stays the primary
log. A node whose preflight cannot reach api.wandb.ai falls back to offline for that job; push those later
from your laptop with `scripts/wandb_sync.sh <ssh-target> <ada-user> [run-name ...]` (the login node cannot
run `wandb`: its wheels need glibc >= 2.28, CentOS 7 has 2.17).

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
configs/train/layout/      research_1gpu (default), nlp_4gpu    configs/eval/*.yaml    eval decoding (paper validation settings)
configs/train/memory/      plan B LoRA (default), A, A'
src/cuts/                  CUTS operator, state, stats channel, vLLM logits processor (verl-independent, CPU-tested)
src/mixed_cuts/            verl integration: scheduler, agent loop hook, trainer, diagnostics, reward, GPQA parser,
                           stability watch, non-thinking check, entry point
src/mc_data/  src/mc_eval/ datasets -> verl parquet schema; pass@1 / pass@16 / maj@16 harness (eval/run_eval.py)
scripts/                   check_env (preflight), compose_config, prefetch, prepare_data, bench_rollout,
                           profile_memory, check_resume, merge_ckpt (+ merge_lora for LoRA runs)
slurm/                     common.sh + sbatch templates (setup, bench, smoke, train, eval, test_resume)
tests/  tests/gpu/         CPU unit tests; GPU tests run inside the smoke job
docs/decisions/            numbered decision records
```

## 10. Developing on a laptop

The Ada login node cannot host this environment (torch wheels need glibc >= 2.28); use a laptop.

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
