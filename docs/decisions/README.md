# Decision records

Short, numbered notes for every choice the research brief did not settle. One file per decision;
add a new number rather than editing an old one when a decision changes, and say why.

| # | Decision |
|---|---|
| [001](001-cuts-proposal-distribution.md) | CUTS is a proposal distribution: no importance correction, actor recomputes `old_log_probs`, a detector guards it |
| [002](002-four-gpu-memory-layout.md) | 4x 2080 Ti layout: TP=4 rollout, FSDP2 CPU offload, effective batch 128 via dynamic micro-batching |
| [003](003-system-prompt-placement.md) | The boxed instruction is the system prompt, identical for training and evaluation |
| [004](004-attention-backend-and-engine-version.md) | `TRITON_ATTN` on sm_75; no V0 engine; Model Runner V1 is forced by the custom logits processor |
| [005](005-run-naming-resume-and-checkpoints.md) | Run naming, durable run dir on /share1, resume, `save_freq`, checkpoint pruning, W&B resume |
| [006](006-fp16-stability-ladder.md) | fp16 watchlist and the mitigation ladder |
| [007](007-cuts-stats-channel.md) | How |S_t| statistics travel from vLLM to the trainer: rank-0 writer, per-step files, one-step lag |
| [008](008-digit-rule-and-filtered-prompts.md) | The boxed-plus-digit validity rule, its training-set filter and the evaluation ceiling |
