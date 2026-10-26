# 018: Ada (2080 Ti) readiness for the decision-017 retrain

Date: 2026-10-26. Status: accepted (user request, 2026-10-26: "change what is needed before training").

## Context

Decision 017 changed the recipe of both arms for a retrain on the H200s. Before the retrain runs on Ada's
RTX 2080 Ti nodes instead, the Ada path was composed against verl v0.9.0 (483b8a0) and checked. Every
configuration composed and passed `make compose-check`, but five things would have gone wrong:

1. `research_1gpu` + `plan_b_lora` still trained LoRA at **lr 1e-5 constant**, the setting both H200 arms
   diverged with (016). 017's calibrated LoRA (lr 2e-6 cosine) existed only GPU-resident
   (`plan_b_lora_calibrated_gpu`), which cannot fit an 11 GB card.
2. Ada run names were `<config>-s<seed>`: `math_grpo-s1` is the September attempt, whose step-1 checkpoint
   (old recipe) is in its durable run dir. With `resume_mode: auto` the new recipe would have resumed it.
3. `save_freq: 1` with `max_actor_ckpt_to_keep: 2` deletes step N when step N+2 is saved: the last good
   checkpoint is gone when a divergence is noticed (on the H200s an ad hoc keeper script saved them).
4. verl starts `reward.num_workers` RewardLoopWorker Ray actors (default 8; `trainer/ppo/v1/trainer_base.py:332`,
   `experimental/reward_loop/reward_loop.py:323-339`), each with `MC_REWARD_WORKERS` math-verify processes,
   on a 30 GB research/low cgroup that sat at 29.4 GB in September; the layout trims every other Ray actor.
5. The stability watch (006) missed both H200 divergences: its entropy rule only covers the first 20 steps
   and `actor/grad_norm` stayed below 2.7. At ~2 h per step on a 2080 Ti an unattended divergence costs days.

## Decisions (one commit each)

1. **`plan_b_lora_calibrated`** (new): `plan_b_lora`'s CPU-offload layout, unchanged memory footprint, with
   017's lr 2e-6 cosine. `configs/ada.env.sh` selects it through `MC_MEMORY_PLAN` (default
   `plan_b_lora_calibrated`; `plan_b_lora` reproduces the earlier runs) for `research_1gpu` and `nlp_1gpu`,
   as `configs/jarvis.env.sh` does on Jarvislabs. 017's default recipe, the full fine-tune at lr 1e-6, is
   `MC_LAYOUT=nlp_4gpu` (`plan_a_fullft_offload`, 4 GPUs on the shared nlp account).
2. **`MC_RUN_TAG=2080ti`** on Ada (`math_grpo-2080ti-s1`), like `h100` and `t4`; and
   **`scripts/check_recipe.py`**, run by `mc_train_main` before training: when the run dir has a checkpoint,
   it compares this submission's composed recipe (lr and schedule, loss aggregation, KL term, LoRA rank and
   alpha, group split, and the step cap under a non-constant schedule) with the newest earlier job's saved
   config (`jobs/*/hydra/.hydra/config.yaml`) and stops the job on a difference unless
   `MC_ALLOW_RECIPE_CHANGE=1`.
3. **`reward.num_workers: 2`** in `research_1gpu` and `kaggle_t4`; `compose_config` refuses more on the 1-GPU
   sm75 layouts.
4. **`mc_keep_checkpoints`** in `slurm/common.sh` (mirror loop every `MC_MIRROR_INTERVAL` s, and stage-out
   before the pruning): every `MC_KEEP_EVERY`-th completed checkpoint (default 10) is hard-linked into
   `checkpoints_kept/`, without the optimizer state (~0.28 GB per LoRA step; for evaluation and
   `scripts/merge_ckpt.sh`, not for resuming).
5. **Policy-drift rule** in `StabilityWatch`: |`actor/ppo_kl`| > 0.02, or entropy > 2x its lowest value so far
   (after 5 steps), on 2 consecutive steps raises `StabilityAbort` (`abort_on_divergence: true`). From the H200
   logs: before diverging |ppo_kl| stayed below 0.0096 in both arms; replayed through the watch the logs alert
   first at Mixed-CUTS step 49 and GRPO step 66 and stop at 50 and 67, with no earlier alert (a test). A stopped
   run resumes on resubmission from its newest checkpoint; set `stability.abort_on_divergence=false` to only
   alert.

## Not done in code (operator steps before the first submission)

* SSH: `ada.iiit.ac.in` no longer accepts SSH (2026-10-26); use `ada-gw1` or `ada-gw2`. The login-node facts of
  decisions 010-012 (CentOS 7, 512 MB virtual-memory cap, `/share1` local) were measured on the old node.
* Pull `main` on Ada; no requirement changed since the last environment build, and the staged parquet files
  stay valid (017's data-filter change was a docstring).
* Smoke tests on a 2080 Ti before the long run (`make sbatch-smoke`, then `smoke_maxlen`): nothing of 017 or of
  the reward changes (pool hardening, symbolic answers) has run in fp16 on sm_75; watch the host-memory peak.
* Step budget: ~2 h 20 min per step means ~10 days per arm on one research GPU, three 4-day submissions. A cap
  must be set with `MC_TOTAL_STEPS` on the first submission and kept (cosine schedule; the recipe check refuses a
  change). Running the second arm on another user's research account halves the wall time.
* Evaluate the base model on Ada as well: the H200 numbers are bf16, Ada evaluations fp16.
