# 009: Checkpoints on node-local scratch, small outputs mirrored to /share1

> **Amended by [010](010-share1-is-login-node-local.md)** (same day, later): `/share1` turned out to be a
> local disk of the login node, invisible from compute nodes, so the durable side of this layout is
> `/home2/<user>/mixed-cuts-data` (25 GB / 300k files), not `/share1`. Everything else here stands.

**Measured on Ada (2026-09-27)**: `/share1/<user>` has a quota of 25,000 MB (hard 27,000 MB) and
3,000 files, not the 100 GB the wiki claimed. One full-fine-tune checkpoint (fp32 master + Adam
for 1.72 B params) is ~21 GB, and saving the next one before pruning the old needs ~42 GB. The
staged model and data take ~4 GB. So checkpoints cannot live on `/share1`. The user chose
node-local scratch over asking for a larger quota.

## Layout (`MC_CHECKPOINT_HOME=scratch`, the default in `configs/ada.env.sh`)

| Where | What | Lifetime |
|---|---|---|
| `/scratch/<user>/mixed-cuts/runs/<run>/` (node-local, 1.8 TB) | the live run dir: `checkpoints/`, `metrics.jsonl`, `phases.jsonl`, `gpu_mem.jsonl`, `cuts_stats/`, `rollout_dumps/`, `wandb/`, `jobs/` | purged by file age (~7 days). Checkpoints are rewritten every step, so the latest survives while a run is active |
| `/share1/<user>/mixed-cuts/runs/<user>/<run>/` (durable) | a mirror of everything except `checkpoints/` (`rsync` every `MC_MIRROR_INTERVAL`, default 10 min, and at exit), plus `node.txt` | permanent (quota permitting) |
| `/share1/<user>/mixed-cuts/{models,data,raw}` | staged model and datasets | permanent |

## Rules that follow

1. **A run resumes only on the node that wrote its checkpoints.** `node.txt` in the durable dir
   records that node; `make sbatch-train CONFIG=... SEED=...` reads it and adds `-w <node>`
   (`MC_PIN_NODE=0` skips the pin). If a job lands on a different node without a checkpoint
   there, `slurm/common.sh` stops with a clear message; `MC_ALLOW_NODE_CHANGE=1` starts over.
   A pinned node that is down or busy means waiting, not silently restarting.
2. **Evaluation reads the HF weights on that node's scratch**: `make sbatch-eval ... NODE=<node>`.
3. **What is safe if the node dies**: everything in the mirror (curves, diagnostics, dumps, W&B).
   What is lost: the checkpoints, i.e. the ability to resume. The run must restart from step 0.
4. **Quota watch**: the preflight reports `/share1` usage and warns above 20 GB or 2,500 files.
   The 3,000-file limit is the tighter one: a uv-managed Python alone exceeded it (so the login
   node keeps its caches under `$HOME/.cache/mixed-cuts`), and the mirror packs the per-step
   directories `cuts_stats/` and `rollout_dumps/` into one `.tar.gz` each instead of copying
   hundreds of small files (`tar xzf` them locally to read).
5. **Switching back**: if the quota is ever raised to >= ~150 GB, `MC_CHECKPOINT_HOME=durable` in
   `configs/local.env.sh` restores the single-directory layout of decision 005; nothing else changes.

Also fixed while measuring:
* `u22` has `MaxMemPerCPU=3000` MB, so `--mem-per-cpu=3G` (3072 MB) would be rejected; every job now
  asks for `3000M` (117 GB per node).
* PyPI's CDN is throttled to ~60-100 KB/s from Ada (login and compute nodes alike; GitHub
  releases too), while the Tsinghua TUNA PyPI mirror does ~4.5 MB/s, `download.pytorch.org`
  6.7 MB/s and `wheels.vllm.ai` 7.6 MB/s. `configs/ada.env.sh` therefore points uv and pip at
  TUNA (`UV_DEFAULT_INDEX`); it mirrors all of PyPI, so the pins resolve identically.
* The login node caps virtual memory at 512 MB per process (hard) and 200 processes. `uv pip install`
  aborted with "memory allocation failed"; the login-node environment is therefore built with plain
  `pip` inside a seeded uv venv, and the Rust download paths (`hf_transfer`, `hf-xet`) are disabled
  there (`configs/ada.env.sh`, `MC_ON_LOGIN_NODE=1`).
