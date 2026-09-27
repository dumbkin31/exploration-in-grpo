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
| `rollout.gpu_memory_utilization` | 0.40 | 0.80 (bench confirms; 0.85 probed) | the actor is in host RAM during rollout: 8.8 GiB = 3.44 weights + ~1 activations/cudagraphs + ~4.3 GB KV (~39k tokens: 6 max-length or ~25 typical sequences) |
| `rollout.max_num_seqs` / `max_num_batched_tokens` | 128 / 8192 | 32 / 4096 | KV-bound; bench picks 16/32/64 |
| `rollout.agent.num_workers`, `data.dataloader_num_workers` | 4, 4 | 2, 2 | 10 cores |
| `ray_kwargs.ray_init.num_cpus` | 40 | 16 (logical) | Ray reserves ~10 for its own actors (TaskRunner 1, worker bundle 3, vLLM server 1, agent workers, TransferQueue); `<= 10` deadlocks placement. The cgroup limits real use |
| `ray_kwargs.ray_init.object_store_memory` | 16 GB | 4 GB | 30 GB host cap |
| `transfer_queue...num_data_storage_units` | 8 | 2 | eight actors cost ~2.4 GB RSS |
| `memory` | plan A | plan B: `offload_policy: true`, `model_dtype: fp32`, `lora.merge: true`, lr 1e-5 | see the table above |
| `OMP_NUM_THREADS`, `MC_REWARD_WORKERS` | 8, 4 | 4, 2 | 10 cores |
| paper hyperparameters (batch 128, mini-batch 32, G=16 8+8, 5000 tokens, T=1, KL 1e-3) | | unchanged | decision D2 stands; only the compute scale and LoRA changed |

Budgets: GPU during the update (vLLM asleep) ~9-9.5 GiB of 11 (one gathered fp16 layer, checkpointed
hidden states 0.66, fp16 logits + log-softmax + grad ~5.3 transient, entropy chunk 1.2); host ~26-27 GB of 30
(FSDP worker ~10.5 steady with 7.2 GB pinned, vLLM ~4.5, driver 2.5, agent/TQ/reward/dataloader workers ~5,
Ray ~1.5, object store 1-2). Risk: verl's non-rmpad path pads each dynamic micro-batch to its longest
member, so a bad mix can exceed the 6144-token budget. **Fallback ladder**: lower `max_num_seqs` -> lower
util -> `actor.use_dynamic_bsz: false` + `ppo_micro_batch_size_per_gpu: 1` (compose-check accepts it) -> as a
last resort `max_response_length` (a deviation from the paper). The smoke job records the per-phase GPU
peaks (`memory_profile.md`) and the cgroup host peak (`jobs/<id>/host_mem_peak.txt`).

## Mechanics

* `MC_LAYOUT` in `configs/ada.env.sh` (default `research_1gpu`; `nlp_4gpu` restores 002) sets the sbatch
  flags (`mc_sbatch_args`: account, QoS, GPUs, CPUs, memory + the driver exclude list) and the Hydra overrides
  (`MC_TRAIN_OVERRIDES`). `make sbatch-*` passes both; the `#SBATCH` headers carry the research defaults.
* `compose-check` asserts `tp == n_gpus_per_node == MC_SLURM_GPUS`, refuses 1 GPU without LoRA + the offload
  policy + merged weights, caps Ray's object store and the worker counts, and the preflight compares the
  visible GPUs with the layout.
* W&B: `WANDB_MODE=online`, `WANDB_DIR=<run dir>` (wandb appends its own `wandb/`), `WANDB_RUN_ID=<run name>`,
  `WANDB_RESUME=allow`; a node that cannot reach api.wandb.ai falls back to offline for that job and
  `scripts/wandb_sync.sh` pushes those runs later.
* Checkpoints stay on node-local scratch with node pinning (009): 7.8 GB per step would not fit the home
  quota. Phase-2 option once the pipeline is proven: `save_lora_only: true` (~0.9 GB) and
  `MC_CHECKPOINT_HOME=durable`, which removes the pin.

## Schedule and what the write-up must say

One step = 2048 sequences x <= 5000 tokens generated on one card (~3 M tokens at 1-2.5k tok/s) plus three
passes over ~3.5 M tokens (old log-probs, ref log-probs, update): **~1-1.5 h per step** until the bench and the
smoke replace the estimate. 100 steps ~ 4.2-6.3 days = two submissions under the 4-day MaxWall (resume on the
pinned node). One GPU per user means the arms run sequentially under one account (~9-13 days for one seed of
both), or in parallel from a second account (the partner's) with its own venv + data in that user's home.
The report states: LoRA r=64 (not full fine-tuning), lr 1e-5, one GPU; both arms identical in every other
respect, so the GRPO vs Mixed-CUTS comparison is unaffected while absolute numbers may differ from Table 1.
