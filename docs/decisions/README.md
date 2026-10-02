# Decision records

Short, numbered notes for every choice the research brief did not settle. One file per decision;
add a new number rather than editing an old one when a decision changes, and say why.

| # | Decision |
|---|---|
| [001](001-cuts-proposal-distribution.md) | CUTS is a proposal distribution: no importance correction, actor recomputes `old_log_probs`, a detector guards it |
| [002](002-four-gpu-memory-layout.md) | 4x 2080 Ti layout: TP=4 rollout, FSDP2 CPU offload, effective batch 128 via dynamic micro-batching |
| [003](003-system-prompt-placement.md) | The boxed instruction is the system prompt, identical for training and evaluation |
| [004](004-attention-backend-and-engine-version.md) | `TRITON_ATTN` on sm_75; no V0 engine; Model Runner V1 is forced by the custom logits processor |
| [005](005-run-naming-resume-and-checkpoints.md) | Run naming, durable run dir (now on /home2, see 010), resume, `save_freq`, checkpoint pruning, W&B resume |
| [006](006-fp16-stability-ladder.md) | fp16 watchlist and the mitigation ladder |
| [007](007-cuts-stats-channel.md) | How |S_t| statistics travel from vLLM to the trainer: rank-0 writer, per-step files, one-step lag |
| [008](008-digit-rule-and-filtered-prompts.md) | The boxed-plus-digit validity rule, its training-set filter and the evaluation ceiling |
| [009](009-scratch-checkpoints-quota.md) | /share1 is 25 GB: checkpoints on node-local scratch, small outputs mirrored, resubmissions pinned to the node |
| [010](010-share1-is-login-node-local.md) | /share1 is a login-node-local disk: staged model/data and the durable run outputs live on /home2 |
| [011](011-mixed-driver-generations.md) | The 2080 Ti nodes run mixed NVIDIA drivers; the cu130 wheels need >= 580: exclude list + a fail-fast guard in every job |
| [012](012-research-qos-one-gpu-lora.md) | `research`/`low` = 1 GPU, 10 CPUs, 32 GB per user: LoRA + FSDP2 offload policy on one card (layout group, `MC_LAYOUT`), W&B online |
| [013](013-jarvislabs-h100.md) | Both arms on Jarvislabs H100s in parallel: bf16, same LoRA, `hardware` config group, `-h100-` run names; the LoRA lr was silently 1e-6 on Ada |
| [014](014-kaggle-t4.md) | Kaggle T4 support (fp16 Ada plan, CUDA 12.9 fallback, runs carried between 12-hour sessions on the HF Hub); use it for smoke tests and evaluations |
| [015](015-length-sorted-microbatches.md) | Padded attention path: micro-batches are length-sorted (patch) instead of workload-balanced, about 4x less padding, same gradient |
| [016](016-h200-divergence-stop-and-evaluate.md) | Both H200 arms diverged (Mixed-CUTS from ~step 50, GRPO from ~step 64): stopped at 72/61 and evaluated at equal steps; GRPO step 40 is the best model |
