> **Superseded as the training platform by [013](013-jarvislabs-h100.md)** (2026-09-29, no Ada access): both arms
> run on Jarvislabs H100s. 013 also found that every LoRA config here composed with lr 1e-6, not the 1e-5 stated
> below (base_grpo.yaml's body overrode the memory plan); fixed there.

# 012: The `research` account gives one GPU per user, so the arms train with LoRA on one 2080 Ti

**Decided 2026-09-27 with the user**: every job runs on SLURM account `research` (QoS `low`) instead of the
shared `nlp` account, and the two arms train with **LoRA on one GPU** instead of the 4-GPU full fine-tune of
[002](002-four-gpu-memory-layout.md). W&B logs **online** from the compute nodes.

## Measured limits (`sacctmgr`, 2026-09-27)

| | `research` / QoS `low` (our account now) | `nlp` / QoS `normal` (shared) |
|---|---|---|
| per user, at any time | cpu=10, gres/gpu=1, mem=32000M | account-wide cpu=120, gpu=12 (everyone in the group) |
| per job | gpu=1, node=1 (`QOSMaxNodePerJobLimit`) | 4 GPUs/node |
| jobs | 5 running / 5 submitted | - |
| MaxWall | 4-00:00:00 (`DenyOnLimit`: longer requests are rejected) | - |
| preemption | `PreemptType=preempt/qos`, `PreemptMode=GANG,SUSPEND`; no QoS lists `low` as a target | - |

`u22` has `MaxMemPerCPU=3000`, so the largest research request is `-c 10 --mem-per-cpu=3000M` = 30 GB,
enforced by the cgroup. Only 7 of the 2080 Ti nodes run a driver the cu130 wheels accept
([011](011-mixed-driver-generations.md)); they are shared with everyone.

## Why full fine-tuning is out and LoRA is in

Plan A keeps fp32 master weights + gradients + Adam state for 1.72 B parameters in host RAM: 27.6 GB before
the ref model, Ray, vLLM and Python exist. It cannot fit 30 GB. LoRA r=64 / alpha=128 on every linear layer
(PEFT `all-linear`, `lm_head` excluded) trains 69.7 M parameters: ~0.85 GB of grads + Adam.

Facts verified in the verl v0.9.0 source before writing the configs:

| Question | Answer (file:line in verl v0.9.0) | Consequence |
|---|---|---|
| second model for the KL reference? | no: `ref_in_actor` when `lora_rank > 0`; ref log-probs come from the actor with `disable_adapter()` (`trainer_base.py:316-323, 1540-1549`, `engine_workers.py:397-419`) | one model; the `ref:` block of plan B was dead config |
| where does the base live? | `fsdp_config.model_dtype` defaults to **fp32** (`workers/config/engine.py:289`); `dtype`/`mixed_precision` only set the compute dtype; LoRA params are cast to the base dtype (`transformer_impl.py:352-365`) | on a 1-GPU world "sharded" = whole: verl's manual `param_offload` would put the 6.9 GB fp32 base on the card for every forward. Only `offload_policy: true` (FSDP2 `CPUOffloadPolicy`, `transformer_impl.py:439-444`) keeps it in pinned host RAM and gathers one layer in fp16 at a time |
| tp=1 on 1 GPU? | `num_replicas = world_size // (tp*dp*pp)` (`llm_server.py:554`); `validate_config` only needs `train_batch_size % n_gpus == 0` | one hybrid replica, nothing assumes > 1 |
| does vLLM leave the card during the update? | with `free_cache_engine` and `lora.merge: true` the hybrid engine sleeps at **level 2** (weights + KV discarded, `vllm_async_server.py:1076-1099`, vLLM `gpu_worker.py:165-177`); merged weights are streamed back after each update | ~1-1.5 GiB of CUDA contexts and cudagraph pools remain |
| host RAM from vLLM? | no `swap_space`, `load_format: dummy` (`vllm_async_server.py:288-313`); V1 has no CPU swap | process RSS only, ~3.5 GB |
| what is a LoRA checkpoint? | the full PEFT state dict (fp32 base + adapters, `fsdp_checkpoint_manager.py:316-362`) unless `actor.checkpoint.save_lora_only` (`trainer/config/config.py:51`), plus AdamW for the LoRA params | ~7.8 GB per step (0.9 GB with `save_lora_only`) |
| can eval load it? | `verl.model_merger` writes the base weights and a separate `lora_adapter/`; it does **not** merge (`base_model_merger.py:311-421`) | `scripts/merge_ckpt.sh` now runs `scripts/merge_lora.py` (PEFT `merge_and_unload`) |
| caveat | the merged tensors are cast to bf16 before vLLM re-casts them to fp16 (`transformer_impl.py` `_merged_lora_per_tensor_param`, `weight_update_utils.py:61`) | one 8-mantissa-bit rounding per step; measured later, one-line patch if it matters |

## The layout (`configs/train/layout/research_1gpu.yaml` + `configs/train/memory/plan_b_lora.yaml`)

| Key | 4-GPU (002) | 1-GPU | Why |
|---|---|---|---|
| `trainer.n_gpus_per_node` / rollout TP | 4 / 4 | 1 / 1 | one replica = the world |
| `rollout.gpu_memory_utilization` | 0.40 | 0.80 | **measured** (bench 2719260, gnode084): 4.74 GiB KV = 44,416 tokens = 7.2 max-length sequences, rollout peak 8.9 GiB. 0.85 starts too (49,360 tokens) but leaves 1.9 GiB beside vLLM; revisit after the smoke's rollout-phase profile |
| `rollout.max_num_seqs` / `max_num_batched_tokens` | 128 / 8192 | 32 / 4096 | **measured**: 125/566/759/896/995 tok/s at concurrency 1/8/16/32/64 (1024-token responses); 4096-token responses saturate at ~285 tok/s from 8 up, so 64 only adds preemption |
| `rollout.agent.num_workers`, `data.dataloader_num_workers` | 4, 4 | 2, 2 | 10 cores |
| `ray_kwargs.ray_init.num_cpus` | 40 | 16 (logical) | Ray reserves ~10 for its own actors (TaskRunner 1, worker bundle 3, vLLM server 1, agent workers, TransferQueue); `<= 10` deadlocks placement. The cgroup limits real use |
| `ray_kwargs.ray_init.object_store_memory` | 16 GB | 4 GB | 30 GB host cap |
| `transfer_queue...num_data_storage_units` | 8 | 2 | eight actors cost ~2.4 GB RSS |
| `memory` | plan A | plan B: `offload_policy: true`, `model_dtype: fp32`, `lora.merge: true`, lr 1e-5 | see the table above |
| `OMP_NUM_THREADS`, `MC_REWARD_WORKERS` | 8, 4 | 4, 2 | 10 cores |
| paper hyperparameters (batch 128, mini-batch 32, G=16 8+8, 5000 tokens, T=1, KL 1e-3) | | unchanged | decision D2 stands; only the compute scale and LoRA changed |

**Measured by the first green smoke run (job 2719378, gnode084, 2 steps, groups of 4, 512-token responses):**
GPU peaks rollout 9.70 GiB, old/ref log-probs 8.86, update **10.05 of 11.26 GiB** (torch: 5.83 allocated, 8.17
reserved -> `expandable_segments` added); host cgroup **peak 29,996 of 30,000 MiB** (median 26.4 GB) -> the
`research_1gpu` layout now runs one agent-loop worker, in-process data loading, one TransferQueue unit and a
2 GB object store, `nlp_1gpu` (60 GB) is the comfortable home for the arms, and `smoke_maxlen` (paper lengths,
`ignore_eos`) is the memory gate before a 4-day submission. W&B synced online
(`anlp-mixed-cuts/mixed-cuts/runs/smoke-s42-2719378`). Step time at the smoke's scale: 0.6 min.

Budgets as planned: GPU during the update (vLLM asleep) ~9-9.5 GiB of 11 (one gathered fp16 layer, checkpointed
hidden states 0.66, fp16 logits + log-softmax + grad ~5.3 transient, entropy chunk 1.2); host ~26-27 GB of 30
(FSDP worker ~10.5 steady with 7.2 GB pinned, vLLM ~4.5, driver 2.5, agent/TQ/reward/dataloader workers ~5,
Ray ~1.5, object store 1-2). Risk: verl's non-rmpad path pads each dynamic micro-batch to its longest
member, so a bad mix can exceed the 6144-token budget. **Fallback ladder**: lower `max_num_seqs` -> lower
util -> `actor.use_dynamic_bsz: false` + `ppo_micro_batch_size_per_gpu: 1` (compose-check accepts it) -> as a
last resort `max_response_length` (a deviation from the paper). The smoke job records the per-phase GPU
peaks (`memory_profile.md`) and the cgroup host peak (`jobs/<id>/host_mem_peak.txt`).

## Attention on Turing (measured 2026-09-28, the decisive memory fact)

PyTorch 2.11's flash and memory-efficient SDPA kernels report *No available kernel* on the 2080 Ti when
`enable_gqa=True` is requested; with the KV heads repeated, the memory-efficient kernel runs at 6,024 tokens
for **+0.09/+0.26 GiB** (causal, forward/forward+backward) and **+0.20/+0.36 GiB** with a boolean mask. The
math kernel costs **+5.2 GiB forward and +8.8 GiB forward+backward per attention call** at that length
(+4.2 GiB at 4,096, +2.4 GiB at 3,072). transformers 5.5.3 requests `enable_gqa` exactly when
`attention_mask is None`, i.e. for single-sequence micro-batches without padding: the long ones. That is
what OOMed the update pass of `smoke_maxlen` job 2719481 (10.8 GiB peak) and what made its log-prob passes
peak at 9.9 GiB. `mixed_cuts.sdpa_patch.install`, run in every Ray worker through
`ray_kwargs.ray_init.runtime_env.worker_process_setup_hook`, makes transformers repeat the KV heads instead.
FlexAttention is not an option on Turing (Triton: *out of resource: shared memory*).

**Result (smoke_maxlen job 2719716, gnode084, 2026-09-28, 16 forced 5,000-token responses, 1 step):** GPU peaks
rollout 9.70 GiB, old-log-prob **2.93**, ref-log-prob **2.93**, update **6.29 GiB** (torch 4.28 allocated / 5.06
reserved); anonymous host memory 18.5 GB of 30; timings rollout 357 s (16 x 5,000 tokens at ~224 tok/s: the
long-context floor), old-log-prob 22 s, ref 20 s, update 64 s, weight sync 17 s. Extrapolated to the real
step (2,048 sequences, mixed lengths): ~70 min rollout + ~11 + 11 min log-prob passes + ~35 min update, i.e.
**~2 h per step**, 100 steps ~ 8-9 days = three 4-day submissions per arm with resume.
The fused log-prob kernel stays: the padded path otherwise materialises the 6k x 152k fp32 logits three
times per sequence (smoke_maxlen job 2719395).

## Mechanics

* `MC_LAYOUT` in `configs/ada.env.sh` (default `research_1gpu`; `nlp_4gpu` restores 002) sets the sbatch
  flags (`mc_sbatch_args`: account, QoS, GPUs, CPUs, memory + the driver exclude list) and the Hydra overrides
  (`MC_TRAIN_OVERRIDES`). `make sbatch-*` passes both; the `#SBATCH` headers carry the research defaults.
  `nlp_1gpu` submits the same 1-GPU LoRA job on the shared `nlp` account: priority 40 instead of 10 and no
  per-user RAM cap (20 CPUs = 60 GB), at the price of counting against the group's 12 GPUs. The account is only
  a submit flag: resume and node pinning do not depend on it, so a run may alternate between the two accounts
  across its 4-day walls, and for short jobs a twin can be submitted under both and the loser cancelled.
* `compose-check` asserts `tp == n_gpus_per_node == MC_SLURM_GPUS`, refuses 1 GPU without LoRA + the offload
  policy + merged weights, caps Ray's object store and the worker counts, and the preflight compares the
  visible GPUs with the layout.
* W&B: `WANDB_MODE=online`, `WANDB_DIR=<run dir>` (wandb appends its own `wandb/`), `WANDB_RUN_ID=<run name>`,
  `WANDB_RESUME=allow`; a node that cannot reach api.wandb.ai falls back to offline for that job and
  `scripts/wandb_sync.sh` pushes those runs later.
* `make sbatch-train` adds `--dependency=singleton`: SLURM runs one job per (user, job name) at a time, so a
  resubmission made before the wall queues behind the live run instead of writing into the same run dir.
  Consequence: an arm is submitted on ONE account (a twin on the other account would sit at `Dependency`
  until the first one ended, unpinned); twins are for the stateless jobs (bench, smoke). Measured: the resume test (2719790 -> 2719803) killed at 15, resumed at 16,
  ran to 30 with one W&B run id; a LoRA checkpoint is 7.3 GB on scratch.
* **Arm 1 (jobs 2719894 and 2719898, 2026-09-28) died twice at the end of a step** with
  `slurmstepd: oom-kill` (host cgroup, 30 GB), inside `checkpoint_manager.update_weights` right after the
  checkpoint save, from ~18 GB of anonymous memory in steady state. The full PEFT state-dict save gathers
  a 6.9 GB fp32 copy on the host and the weight sync stacks its merge and 2 GB bucket copies on top.
  Since then: `checkpoint.save_lora_only: true` (adapters + optimizer, ~0.9 GB),
  `rollout.checkpoint_engine.update_weights_bucket_megabytes: 512`, and **`MC_CHECKPOINT_HOME=durable` for
  the LoRA layouts**: checkpoints live in the run dir on `/home2`, resumes and evaluations run on any good
  node, no `-w` pin (`compose_config.py` refuses a 1-GPU layout without `save_lora_only`).
  `scripts/merge_ckpt.sh` rebuilds the PEFT adapter from the LoRA-only `.pt` and merges it into the base
  model (`merge_lora.py --verl-actor-dir`); verl's own merger asserts on the missing base keys.
  Step time measured on the arm: rollout 59 min, old log-probs 16.5, ref 15.3, update 49.6 ->
  **~2 h 20 min per step** (100 steps ~ 10 days: three 4-day submissions per arm).
* Full-fine-tune runs (`MC_LAYOUT=nlp_4gpu`) keep node-local scratch with node pinning (009).
* **Host-memory margin after the fix (resume test 2720711 -> 2720815, gnode087, 2026-09-28: `RESUME CHECK OK`,
  810 MB checkpoints on /home2).** The sidecar's new `shmem` column shows **11.2 GB of shared memory from the
  first 20 s of `init_model`**, constant for the whole job, next to an anonymous peak of 18.5 GB: together
  **29,730 of 30,000 MiB**. The failed arm's anonymous memory at full scale peaked at the same 18.6 GB, so
  batch size barely moves the host budget; the margin is ~270 MiB either way. Source of the shared memory
  (verified): verl builds FSDP2's policy as `CPUOffloadPolicy(pin_memory=True)` unconditionally
  (`transformer_impl.py:443`), and torch 2.11's `CachingHostAllocator` rounds every pinned allocation up to a
  power of two and caches freed blocks (`ATen/core/CachingHostAllocator.h:302`). Pinned host memory is a
  shared mapping, so it counts as `shmem`, not `anon`. The fp32 base is 6.9 GB; the power-of-two rounding of
  the embedding (1.24 -> 2.15 GB) and the MLP weights (50 -> 67 MB each) accounts for ~9.2 GB, the rest is
  presumably cached free blocks from `fsdp2_load_full_state_dict`'s GPU round trip (to be confirmed with
  `torch.cuda.host_memory_stats()`). Candidate fixes, not applied: release the pinned cache after model init
  (~2 GB), or unpinned offload (~4.3 GB, slower host-to-device copies).
* **Arms may now be queued on both accounts.** The one-account rule above existed because an unpinned twin
  could not reach scratch checkpoints. With durable checkpoints, a same-name twin under `MC_LAYOUT=nlp_1gpu`
  (60 GB host memory) held back by `--dependency=singleton` is a valid automatic continuation: it starts only
  after the running job ends and resumes from the last checkpoint on any node.
* **Third OOM, and its real cause (job 2721833, gnode061, 2026-09-29 01:45).** The arm completed step 1 at
  full scale (rollout, both log-prob passes, update; the per-process sidecar showed the FSDP worker holding
  all 11.2 GB of pinned shared memory and 5.6 GB private, so no stray copy of the weights), saved the
  810 MB step-1 checkpoint, and was OOM-killed ~48 s later inside the weight sync. Steady host memory during
  the step was 29.4 of 30 GB; at the sync the job had ~2.5 GB of headroom. Verified in verl v0.9.0:
  `FSDPEngine._merged_lora_per_tensor_param` (`transformer_impl.py:1048`) streams merged weights inside
  `merged_lora_context(self.module, backup_adapters=True)`, whose `backup_base_model_weights` clones every
  non-LoRA parameter to a fresh CPU tensor: 6.9 GB fp32 in one go, with no config switch. The small-batch
  resume test survived only because its memory at sync time was ~4 GB lower.
  **Fix (user-approved hook, `mixed_cuts.lora_sync_patch`):** replace `transformer_impl`'s reference to the
  context with one that passes `backup_adapters=False`. verl's own fallback then runs: after streaming,
  `fsdp_merge_unmerge(do_merge=False)` calls PEFT's `LoraLayer.unmerge`, which subtracts the same
  `scaling * B @ A` that `merge` added. Cost: at most ~1 ulp of fp32 rounding per step in the frozen base
  (measured round trip over 100 cycles: < 1e-6 relative; fp16 sampling resolves ~1e-3). The wrapper raises
  if any LoRA layer is still merged after a sync, so a failed unmerge cannot silently double-count the
  adapter. It is the only caller of the context in verl; `collect_merged_lora_params`, the other user of the
  backup, has no callers. The patch installs lazily (a one-shot import finder on the engine module), because
  importing the FSDP engine in every Ray worker would itself cost host memory. `MC_LORA_SYNC_BACKUP=1`
  restores verl's behaviour. Both arms run with it, so the comparison stays like for like. Ray takes one
  setup hook, so `mixed_cuts.worker_hooks.install` runs this and the SDPA patch; compose-check requires it.
  A test asserts verl's source still hardcodes the backup, so a verl upgrade re-opens this decision.

## Schedule and what the write-up must say

One step = 2048 sequences x <= 5000 tokens generated on one card plus three passes over the same tokens
(old log-probs, ref log-probs, update). With the measured decode rates (~900 tok/s for short answers, ~285
tok/s for the long tail) a realistic MATH length mix gives ~60-90 min of rollout per step, so **~1.5-2 h per
step** until the smoke's `phases.jsonl` replaces the estimate. 100 steps ~ 4.2-6.3 days = two submissions under the 4-day MaxWall (resume on the
pinned node). One GPU per user means the arms run sequentially under one account (~9-13 days for one seed of
both), or in parallel from a second account (the partner's) with its own venv + data in that user's home.
The report states: LoRA r=64 (not full fine-tuning), lr 1e-5, one GPU; both arms identical in every other
respect, so the GRPO vs Mixed-CUTS comparison is unaffected while absolute numbers may differ from Table 1.
