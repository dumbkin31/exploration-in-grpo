# 010: /share1 is a local disk of the login node; staging and durable outputs live on /home2

**Measured on Ada, 2026-09-27 (evening)**, by the first environment-build job (2719025, gnode084) and a
CPU-only `srun` diagnostic (2719035, gnode043):

* On the login node `/share1` is `/dev/sdg1`, a **local ext4 disk** (95 TB, `mount` shows no NFS export).
* On the compute nodes `ls /share1` says *No such file or directory*. They mount only `/home` and
  `/home2` (NFS 4.2 from 172.16.0.3; a 456 MB file reads at 118 MB/s) and the node-local `/scratch`.
* There is no ssh key pair on Ada, so a compute node cannot `rsync ada:/share1/...` either
  (`Permission denied (publickey,password,hostbased)`), and `autofs` is inactive.

So the layout of [009](009-scratch-checkpoints-quota.md), which staged the model and datasets on `/share1`
and mirrored every run's small outputs there, could never have worked from a job: the setup job's
preflight failed its storage check and skipped the data build.

## Options considered

| | Cost | Why not / why |
|---|---|---|
| **A. `/home2/<user>/mixed-cuts-data` as `MC_STAGE_ROOT`** (chosen) | 5.2 GB of the 25 GB home quota; nothing new to install or authorise | visible on every node at NFS speed, same file system the venv already lives on, one env-var default changed |
| B. keep `/share1`, create an ssh key on Ada and `rsync ada:/share1/...` at job start | a new key in `~/.ssh/authorized_keys`, a per-job dependency on the login node, a 5 GB copy per job and node | more moving parts for the same bytes; the key is an account-security change the owner should make, not a job script |
| C. download from the Hub into `/scratch` on each node | ~30 min per new node at 3 MB/s | slow, and the raw datasets would still need a durable home for the parquet build |

## Layout after this decision (`configs/ada.env.sh`)

| Path | What | Quota |
|---|---|---|
| `/home2/<user>/mixed-cuts/` | the repo and its `.venv` (~9.6 GB) | shared 25 GB / 300k files |
| `/home2/<user>/mixed-cuts-data/` = `MC_STAGE_ROOT` | `models/`, `raw/`, `data/` (parquet), `MANIFEST.json`, `runs/<run>/` (the durable mirror of 009, now without the `<user>` level), `jobs/` (setup/data preflight facts) | same 25 GB |
| `/scratch/<user>/mixed-cuts/` | unchanged: live run dirs with checkpoints, stage-in copies, caches | 1.8 TB, purged |
| `/share1/<user>/` | unused. The first prefetch (5.2 GB) is still there as a backup copy | 25 GB / 3,000 files, login node only |

Budget on the home quota after the copy: 15.3 GB used (venv 9.6, staged 5.2, caches 0.3), ~10 GB left
for parquet files and the durable outputs of all runs. The mirror already excludes checkpoints and packs
`cuts_stats/` and `rollout_dumps/` into gzip archives (roughly 5x smaller for jsonl); the preflight now
reports home usage and warns above 20 GB or 250k files. If the budget gets tight, delete old runs'
`rollout_dumps.tar.gz` first (they are for manual reading; W&B and `metrics.jsonl` hold the curves).

## What changed in the repo

* `MC_STAGE_ROOT` defaults to `${HOME}/mixed-cuts-data`; `MC_RUNS_DIR` to `${MC_STAGE_ROOT}/runs`.
  Override either in `configs/local.env.sh` if a different home ever appears.
* `scripts/check_env.py`: the storage check reports home usage in GiB and files instead of `/share1`
  usage; `math_verify`'s version is read from the installed distribution (it has no `__version__`, which
  the setup job reported as a false FAIL); the SLURM line labels `SLURM_JOB_GPUS` as GPU ids, not gres.
* `slurm/data.sbatch` runs the full preflight (with the staged-data check) after building the parquet
  files, so the first green preflight is on record in `mixed-cuts-data/jobs/`.
* `scripts/wandb_sync.sh` pulls from `/home2/<user>/mixed-cuts-data/runs` by default.
* `HF_HUB_ENABLE_HF_TRANSFER` is no longer set anywhere: `hf_transfer` is not installed, and
  huggingface_hub 1.x warns when the variable is present at all.

Rules 1-3 and 5 of 009 (node pinning, evaluation on the checkpoint node, what survives a lost node, the
switch back to a single durable run dir) are unchanged; rule 4 now watches the home quota.
