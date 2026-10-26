# 017: Recipe changes after the H200 divergence

Date: 2026-10-26. Status: accepted (user brief, 2026-10-26).

## Context

Both H200 arms diverged (decision 016). The Mixed-CUTS logs show the order of events: entropy fell from 0.17
(step 5) to 0.03 (step 40) while the KL to the reference crept from 0.01 to 0.17 (step 48); then `actor/ppo_kl`,
the drift from pi_old inside a single step, jumped from 0.009 to 1.5 over steps 48-54, and entropy rose to 6.7
(gibberish, 94% of responses truncated). `actor/grad_norm` never exceeded 2.7. GRPO showed the same pattern
over steps 63-72.

Four changes apply to BOTH arms before retraining, one commit each so they can be ablated. Unchanged: the CUTS
sampler (K=5, delta=0.03, T_warm=5), no importance correction for CUTS rollouts (001), G=16 (16+0 / 8+8),
128 prompts, mini-batch 32, 5000-token responses, temperature 1.0. Every verl reference below is v0.9.0
(commit 483b8a0, the pin in `requirements/lock.txt`).

## 1. Per-rollout loss normalisation: `loss_agg_mode: seq-mean-token-mean`

verl's default `token-mean` divides the summed token losses of a mini-batch by its total token count, so a long
rollout weighs more than a short one. `seq-mean-token-mean` averages each rollout over its own length, then
averages the rollouts: the 1/|o_i| of the CUTS paper's Eq. 2 (as stated in the brief).

* verl: `trainer/ppo/core_algos.py:1194` (`agg_loss`). The divisor is `ppo_mini_batch_size x rollout.n` = 512
  rollouts (`trainer/ppo/v1/trainer_base.py:1696`), so the mean is exact under dynamic micro-batching. The
  same mode aggregates the KL term (`workers/utils/losses.py:136`). `ActorConfig.__post_init__` rejects unknown
  modes.
* `make compose-check` refuses any other mode.
* Watch item: per-rollout normalisation penalises a long wrong answer less per token than a short one (the
  length bias described by Dr. GRPO); watch `response_length/mean`.
