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

## 2. KL estimator: `kl_loss_type: low_var_kl+` (use_kl_loss true, coefficient 1e-3 unchanged)

The brief proposed k1 (`kl`) instead of k3 (`low_var_kl`), because k3 contains pi_ref/pi_theta. Checked in verl
first (`trainer/ppo/core_algos.py:2187-2245`), and numerically with that code on a toy policy:

* k3's value cannot blow up: verl clamps the log-ratio to +-20 and the value to <= 10. Its per-token
  gradient on a sampled token is (1 - pi_ref/pi_theta) * grad log pi_theta: at most 1 where the policy
  over-weights a token, and exactly 0 once log pi_theta - log pi_ref exceeds ~11 (the value clamp). The
  pull towards the reference fades where the policy has drifted furthest. On-policy its expected gradient
  is the gradient of the forward KL (cosine 0.55 with the reverse-KL gradient on the toy policy).
* k1 as a loss term has gradient grad log pi_theta(token), whose expectation under on-policy sampling is
  zero (6e-8 on the toy policy): it adds noise but no pull towards the reference.
* `low_var_kl+` (the `+` suffix, verl `docs/algo/grpo.md:49`, tested in verl
  `tests/trainer/ppo/test_core_algos_on_cpu.py:343`) keeps the k3 value in the forward pass, so
  `actor/kl_loss` stays comparable with the earlier runs, and back-propagates the k2 gradient
  (log pi_theta - log pi_ref) * grad log pi_theta through a straight-through estimator: the exact gradient
  of KL(pi_theta || pi_ref) for on-policy samples (cosine 1.000 on the toy policy), proportional to the
  log-ratio, no probability ratio, no clamp.

User decision (2026-10-26): `low_var_kl+`. The logged value is still the paper's low-variance estimator; the
gradient is not (a deviation from Table 3 to state in the write-up). verl does not validate `kl_loss_type`
when the config loads (a misspelling raises NotImplementedError at the first update), so `make compose-check`
pins the exact string.

The H200 logs do not support a k3 blow-up as the trigger: `actor/grad_norm` stayed below 2.7, and the first
divergence signal was `actor/ppo_kl` (pi_theta vs pi_old inside one step) while `actor/kl_loss` was still ~0.2.
