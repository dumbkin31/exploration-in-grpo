# 005: Run naming, durable run dir, resume, save frequency, W&B

Multi-hundred-step runs will be killed and resubmitted. Everything a resumed job needs must be
in one durable place, found by name, not by job id.

* **Run name** `MC_RUN_NAME = <config>-s<seed>` (e.g. `math_mixed_cuts-s1`); override it to keep
  several runs of the same config. `make sbatch-train CONFIG=... SEED=...` sets it.
* **Run dir** `MC_RUN_DIR = /share1/<user>/mixed-cuts/runs/<user>/<run name>` (durable) holds
  `checkpoints/`, `metrics.jsonl`, `phases.jsonl`, `gpu_mem.jsonl`, `cuts_stats/`,
  `rollout_dumps/`, `wandb/` and `jobs/<slurm job id>/` (per-submission logs, preflight
  report, `env_facts.json`). Node-local `/scratch` only holds staged model/data and caches.
  `scripts/check_env.py` FAILS when the durable dir is not writable.
* **Resume**: `trainer.resume_mode: auto` makes verl read
  `checkpoints/latest_checkpointed_iteration.txt` and restore actor, optimizer, LR scheduler,
  RNG and dataloader state (`trainer_base.py:795-835`). A fresh dir starts from step 0. verl
  cannot checkpoint on SIGTERM: the `--signal=B:SIGUSR1@300` grace only flushes logs; the resume
  point is always the last periodic checkpoint.
* **`save_freq`**: in steps. One step is plausibly 40-90 min on this hardware, so the default
  is `1`; `scripts/profile_memory.py --report` prints the measured step time and the
  `save_freq` that corresponds to ~30 min. Each FSDP checkpoint is ~27 GB.
* **Pruning**: verl's `max_actor_ckpt_to_keep` never prunes checkpoints saved before a restart,
  so `slurm/common.sh::mc_prune_checkpoints` keeps the tracked step plus one older one.
* **W&B**: `WANDB_RUN_ID = run name`, `WANDB_RESUME = allow`, offline mode; one set of curves
  across resubmissions. Steps re-logged after a kill (the step between the last checkpoint and
  the kill) are dropped by W&B's monotonic-step rule; `metrics.jsonl` (append-only, primary) has
  them and is what ACR_100 and the stability watch rebuild from.
* **Verification**: `sbatch slurm/test_resume.sbatch` runs the smoke config for 30 steps with
  `save_freq 5`, sends the batch shell a real SIGUSR1 when step 15 is logged (the same path
  SLURM uses), resubmits itself with `--dependency=afterany`, and `scripts/check_resume.py`
  asserts the second job logged step 16 first, one W&B run id, and a final tracked step of 30.
* **Traps**: the sbatch scripts `exec` the launcher so the batch shell (the PID SLURM signals)
  is the one holding the traps.
