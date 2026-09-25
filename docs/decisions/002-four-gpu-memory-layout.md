# 002: Four-GPU memory layout (Task A)

**Hardware**: one node, 4x RTX 2080 Ti (11 GiB, sm_75, no bf16, no FlashAttention), 40 cores,
128 GB RAM (120 GB requested). Model: Qwen3-1.7B, 1.72 B parameters, fp16 weights 3.44 GB, KV
cache 112 KB/token (2 x 28 layers x 8 KV heads x 128 x 2 bytes).

## Rollout: one vLLM engine, tensor parallel 4

`rollout.tensor_model_parallel_size: 4` gives one replica over the four cards
(`llm_server.py:528-564`: `num_replicas = world_size // tp`). Weights shard to 0.86 GB per card;
at `gpu_memory_utilization: 0.40` about 3.5 GB per card is KV cache: ~125k tokens in total, i.e.
~20 concurrent 6k-token sequences or ~80 at 1.5k. During training the engine sleeps at level 2
(`_sleep_hybrid`: full-weight training -> level 2), which frees weights and KV; weights are
re-synced after every optimizer step.

Alternative kept as a one-line knob: `tensor_model_parallel_size: 1` gives four data-parallel
replicas, each holding the full 3.44 GB of weights and ~4 GB KV at 0.70 utilisation. TP=4 over
PCIe (no NVLink) costs latency per token; DP=4 costs memory per card. `make sbatch-bench`
measures both; the config default follows the brief (TP=4).

## Training: FSDP2 sharded over 4 ranks with CPU offload

verl v0.9.0 has two offload mechanisms and only one keeps the optimizer state off the GPU
during `optimizer.step` (`transformer_impl.py:440-444`):

| Flags | Where params / grads / Adam live during the update |
|---|---|
| `strategy: fsdp2` + `fsdp_config.offload_policy: true` (**plan A, default**) | pinned host RAM (torch FSDP2 `CPUOffloadPolicy`); the step runs on the CPU |
| `strategy: fsdp2` + `param_offload: true`, `optimizer_offload: true` (plan A', tuning fallback) | moved back to the GPUs for the update: fp32 master + grads + Adam = 27.6 GB / 4 = ~7 GB per card, before activations |

The brief's words "param_offload and optimizer_offload to CPU" therefore map to plan A.

Per-GPU budget in plan A: one all-gathered fp16 layer (< 0.2 GB) + activations of a 6144-token
micro-batch with gradient checkpointing (~2.5 GB) + fp16 logits 6144 x 151,936 (1.9 GB) +
chunked fp32 log-softmax/entropy (~0.6 GB) + CUDA context -> ~5-6 GB peak. The reference model
uses the same policy, forward only. Host RAM: actor 27.6 GB + ref 6.9 GB pinned + 16 GB Ray
object store + vLLM processes -> well inside 120 GB. `scripts/profile_memory.py` measures the
real peaks per phase; `gpu_memory_utilization` can be raised once they are known.

## Effective batch 128 without a separate accumulation knob

In verl the effective batch is `data.train_batch_size` (128 prompts x 16 rollouts = 2048
sequences). Rollouts stream into TransferQueue on the CPU, so generating 2048 sequences is not a
GPU-memory problem; time is the constraint. `actor.ppo_mini_batch_size: 32` gives four optimizer
steps per training step, and inside each mini-batch `use_dynamic_bsz` with
`ppo_max_token_len_per_gpu: 6144` packs sequences into micro-batches of at most 6144 tokens
(the longest possible sequence is 1024 + 5000 = 6024, and without remove-padding a micro-batch
is padded to its longest member). Gradients accumulate across those micro-batches: this is the
"gradient accumulation" of the brief, 32 x 16 / 4 = 128 sequences per GPU per optimizer step.

## Consequences

* One step generates up to 2048 x 5000 tokens on a single TP=4 engine with Triton attention:
  plausibly 40-90 minutes. `save_freq: 1` and resume-on-resubmit (005) follow from that.
* `max_num_seqs: 128` and `max_num_batched_tokens: 8192` are starting points for the bench.
* LoRA (plan B) stays a documented fallback only; it weakens the reproduction claim.
