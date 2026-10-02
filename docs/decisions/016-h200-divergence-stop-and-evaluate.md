# 016: Stop both H200 arms after they diverged; evaluate at equal steps

Date: 2026-10-01. Status: accepted (user decision, 20:00 UTC).

## Context

Both arms trained on 2x H200 (decision 013, with the length-sorted micro-batches of decision 015). Validation
(100 MATH-500 problems, mean@4) peaked at step 30 for Mixed-CUTS (73.5%) and step 40 for GRPO (77.3%). Mixed-CUTS
then diverged from about step 48 and collapsed by step 56 (entropy 6.8, 94% of responses truncated, validation
0.3% at step 60, 34 minutes per step). GRPO followed from about step 64 (KL 0.15 to 1.05, truncation 37%,
validation 69.5% at step 70, 17 minutes per step). Two to three hours of the INR 5,880 credit were left.

## Decision

* Stop both arms (GRPO at step 72, Mixed-CUTS at step 61) instead of training on towards step 100: further
  steps could not beat the earlier checkpoints, and the credit had to cover evaluation.
* Evaluate on both GPUs in parallel, at equal steps, from checkpoints preserved by a keeper script on the
  instance (verl keeps model weights only for the last two checkpoints, `trainer.max_actor_ckpt_to_keep: 2`):
  the base model, both arms at step 40, both at step 50 (and step 22, where both still had weights).
* Report GRPO step 40 as the best model, and the equal-step comparison at step 40 as the main result.

## Consequences

* Step-22 checkpoints were pruned by the launcher's cleanup when the runs were stopped, so those two
  evaluations were skipped. Keep copies of every checkpoint you intend to evaluate before stopping a run.
* GRPO step 50 is not evaluated: `check_non_thinking` aborts evaluation when any response contains a think tag
  (9 of 8,000 here, with all prompts in non-thinking mode). It needs a tolerance (a fraction, reported as a
  metric) before such checkpoints can be scored.
* The stability watch (decision 006) did not flag either divergence: its entropy rule only covers collapse in
  the first 20 steps. A rule on KL to the reference (rising for many steps, then a jump) or on entropy rising
  far above its running level would have caught both runs 5 to 10 steps before validation fell.
* The shared failure mode (KL creeps, then explodes) suggests the shared recipe rather than CUTS alone. A plausible
  but untested cause is the bf16 rollout/training probability mismatch that truncated importance sampling
  corrects (turned off by decision 001); full fine-tuning at lr 1e-6 (the paper) versus LoRA at 1e-5 is the other
  obvious difference.
* Results: `results/h200-2026-10-01/`. Checkpoints and sample-level outputs: Hugging Face `lokola13/mixed-cuts-h200-runs`
  (private). W&B: steps lost to the spot pauses and the stop are listed in each run's notes.
