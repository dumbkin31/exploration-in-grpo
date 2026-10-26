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

## 3. Full fine-tuning on one H200, with a calibrated LoRA fallback

Two new memory plans; `plan_b_lora_gpu` (LoRA, lr 1e-5 constant) stays to reproduce the H200 runs.

| | `plan_a_fullft_gpu` (default) | `plan_b_lora_calibrated_gpu` |
|---|---|---|
| `model.lora_rank` / `lora_alpha` / `target_modules` | 0 (full fine-tune) | 64 / 128 / all-linear |
| `actor.optim.lr` | 1.0e-6 | 2.0e-6 |
| `actor.optim.lr_scheduler_type` | constant | cosine (no warmup, to 0 at `total_training_steps`) |
| offload (actor + reference) | none | none |
| checkpoint | model + Adam, ~21 GB | adapters + Adam, ~0.9 GB |
| reference model | separate fp32 copy (6.9 GB) | the actor with the adapter disabled |

verl keys checked in v0.9.0: `model.lora_rank`/`lora_alpha`/`target_modules` (`config/model/hf_model.yaml:42-49`,
consumed by `workers/engine/fsdp/transformer_impl.py:143,342`; `lora_rank: 0` disables LoRA),
`actor.optim.lr_scheduler_type` in {constant, cosine} (`config/optim/fsdp.yaml:39`, asserted in
`workers/config/optimizer.py:123`; `warmup_style` is its deprecated name). The scheduler steps once per
training step (`workers/engine_workers.py`, `update_lr_scheduler`), over `trainer.total_training_steps`
(`trainer/ppo/v1/trainer_base.py:712-728`).

**Selection.** The Jarvislabs launcher passes `memory=...` on the command line (`MC_TRAIN_OVERRIDES`), which
would override any default in the YAML. `configs/jarvis.env.sh` therefore builds that override from
`MC_MEMORY_PLAN` (default `plan_a_fullft_gpu`, user decision 2026-10-26). Both arms always get the same plan.

**Memory on one H200 (141 GB), full fine-tune.** The LoRA arms peaked at 16.1 GB allocated in the update
(fp32 base 6.9 GB + adapter state + ~8 GB activations at 32k-token micro-batches with gradient
checkpointing). Full fine-tuning adds fp32 grads (6.9 GB, update only) and Adam (13.8 GB) and a separate fp32
reference (6.9 GB): rollout ~121 GB (vLLM 0.60 x 141 = 85 GB beside ~35 GB of actor and reference), update
~43 GB with vLLM asleep, and verl empties the CUDA cache after each update (`engine_workers.py:761`) before
vLLM wakes up. No OOM expected; micro-batching, gradient checkpointing and the batch sizes are unchanged. It
does not fit an 80 GB H100 at this layout's 0.60 (48 + 35 + 2 > 80): use the LoRA plan there.

**Disk, the real risk.** `save_freq: 1` and `max_actor_ckpt_to_keep: 2` with ~21 GB checkpoints in the durable
run dir: ~42 GB per arm, ~84 GB for both, plus a transient third checkpoint while saving; the instance's
`/home` had 73 GiB free at the last H200 attempt. Free disk, or lower `max_actor_ckpt_to_keep`/raise
`save_freq`, before a full-fine-tune run. `scripts/merge_ckpt.sh` already converts full-FT checkpoints with
verl's model merger.

**Run names.** `resume_mode: auto` resumes whatever checkpoints a run dir holds, and the instance's
`configs/local.env.sh` sets `MC_RUN_TAG=h200`: relaunching seed 1 would resume the diverged runs. Start the
new recipe under a new `MC_RUN_TAG` (or seed).

**Recipe guard.** `slurm/common.sh::mc_check_recipe` enforces this before the trainer starts. A run's first
start writes `<run dir>/recipe.txt`: `MC_TRAIN_OVERRIDES` and the composed `model.lora_rank`, `actor.optim.lr`,
`actor.optim.lr_scheduler_type`, `trainer.total_training_steps`, `actor.loss_agg_mode` and `actor.kl_loss_type`
(one `compose_config.py` call with several `--select`). A resume (a checkpoint exists) under a different recipe exits 7 and prints both; so
does a run dir with checkpoints but no `recipe.txt`, i.e. every run started before the guard. The H200 runs
are such dirs, and no launch after this decision reproduces their recipe (`token-mean`, `low_var_kl`).
`MC_ALLOW_RECIPE_CHANGE=1` resumes anyway and keeps the old file as `recipe.until-step-<N>.txt`. `jarvis/train.sh`
does not retry a refusal; `scripts/hub_sync.py` carries `recipe.txt` between Kaggle sessions.

The cosine schedule of the LoRA plan decays to 0 at `trainer.total_training_steps`: capping a run with
`MC_TOTAL_STEPS` shortens it, and changing the cap on a resume makes the learning rate jump, so the guard
refuses a new cap on a resume. It does so under a constant schedule too, where the cap only moves the last
step: pass `MC_ALLOW_RECIPE_CHANGE=1` there.

## 4. Symbolic answer grading: a non-empty final `\boxed{}` is valid; no digit required

`mixed_cuts.reward.extract_answer` returns `valid = len(pred) > 0` instead of "contains a digit". Answers
such as `\frac{3}{4}` or `2\sqrt{5}` already passed (they contain digits); the change admits digit-less
answers (`\pi`, `\infty`, `p - q`, `\cot x`, `\text{east}`, `\text{(C)}`), which math-verify now grades. An
answer it cannot parse, or that times out, still scores 0 (the subprocess call is wrapped, `score_math`).
This replaces the paper's Appendix B.4 validity rule of decision 008 for training AND evaluation (user
decision 2026-10-26: the training reward and eval correctness stay one function). Unchanged: the training
set filter of decision 008 (`scripts/prepare_data.py` still drops the 116 digit-less MATH ground truths:
7,382 training problems), and GPQA's letter parser.

**Re-scored MATH-500 evaluations of the H200 runs** (pass@1, mean over 16 samples; only the 11 digit-less
problems changed, no answer to a numeric problem flipped): base 70.19 -> 71.88, GRPO step 40 78.11 -> 80.08,
Mixed-CUTS step 40 71.51 -> 73.20, Mixed-CUTS step 50 67.36 -> 69.01. Compare new runs with these numbers,
not with the ones in `results/h200-2026-10-01/`. AIME and AMC ground truths all contain digits. The
evaluation's `digit_rule_ceiling` field is now only the share of digit-containing ground truths, not a
ceiling.

**Could the training filter be dropped?** Technically yes: all 11 digit-less MATH-500 ground truths
(names, directions, letter choices, symbolic expressions; the same kinds as the 116 filtered training
answers) parse and verify against themselves, and the four models above scored 5-16 of 16 on each under the
new grader with plausible verdicts (`\cot x` = `\frac{\cos x}{\sin x}`; `-\pi` != `\pi`). Caveats: the 116
training answers themselves were not checked (the raw dataset is not on this machine; run
`reward._math_verify_equal(gt, gt)` over them first); letter answers are guessable and free-text verdicts
are strict (`\text{(E) Hyperbola}` scores 0 against `\text{(E)}`); and dropping the filter changes the data
(and its seeded order) relative to the H200 runs. It adds 1.5% of the prompts, so keeping the filter for this
retrain keeps the four changes the only differences.
