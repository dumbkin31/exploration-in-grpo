# 011: Mixed NVIDIA driver generations on the 2080 Ti nodes

**Measured on Ada, 2026-09-27.** The data job (2719042) landed on gnode054 and its preflight died in
`torch.cuda` init: *"The NVIDIA driver on your system is too old (found version 12080)"*. The setup job
had run on gnode084 with driver 580.178.04 (CUDA 13.0), which is what the pins were built for. A census
of `/proc/driver/nvidia/version` over every 2080 Ti node with a free CPU (25 of 44; the other 19 were
fully allocated, drained or reserved), one CPU-only `srun` each:

| Driver | Nodes | Runs the cu130 wheels |
|---|---|---|
| 580.178.04 | gnode065 068 078 081 084 087 | yes |
| 595.91.07 | gnode070 | yes |
| 570.211.01 (CUDA 12.8) | gnode043 050 054 056 072 073 079 080 082 085 090 091 | **no** |
| no kernel module loaded | gnode066 076 088 089 | **no** |
| probe timed out | gnode052 057 | unknown |
| 580.178.04 but **CUDA cannot initialise** (`nvidia-smi` healthy, modules and `/dev` nodes present, `torch.cuda.init()` -> "CUDA unknown error" for every process; found by the first bench job 2719206) | gnode065 | **no** (node fault; reported to the admins) |

The generation does not follow any SLURM feature: `phase3` nodes are on both sides, and so are plain
`2080ti` nodes. So `-C` cannot select it.

## Why not change the pins instead

* vLLM 0.24.0 publishes exactly two GPU wheels: the PyPI default (cu130) and `+cu129` on the GitHub
  release. Both need a driver >= 575/580. There is no cu128 build, and `wheels.vllm.ai` has no
  per-CUDA index for releases.
* Building vLLM 0.24.0 from source against CUDA 12.8 on a 40-core node is hours of work per rebuild and
  a second environment to keep in step with verl; not worth it for a driver rollout the admins are
  visibly doing (the 580 nodes are the upgraded ones).
* An older vLLM with cu128 wheels (0.10/0.11) would mean re-verifying every interface this project
  depends on (V1 logits processor, `processed_logprobs`, the agent-loop hook) against a different tree.
  Out of scope; see decision 004 and the phase-1 brief.
* CUDA forward-compatibility packages do not cover GeForce cards.

## Decision

1. **Exclude the known-bad nodes at submit time.** `configs/ada.env.sh` carries the measured list in
   `MC_SLURM_EXCLUDE` and the helper `mc_sbatch_exclude`; every `make sbatch-*` target passes its output
   (`-x <nodes>`) to `sbatch`. Submitting `sbatch slurm/x.sbatch` by hand skips this: use make.
2. **Fail fast on any other bad node.** `mc_check_driver` (in `slurm/common.sh`) runs first thing in
   `mc_job_init`, and in the setup and data jobs: it reads `/proc/driver/nvidia/version`, exits with
   code 6 if the major version is below `MC_MIN_DRIVER_MAJOR` (580) or the module is missing, and appends
   the node to `MC_BAD_NODES_FILE` (`<stage root>/bad_nodes.txt`, durable). `mc_sbatch_exclude` merges
   that file into the exclude list, so the next submission of the same make command avoids the node.
   The guard runs before the run dir or `node.txt` is written, so a bad landing leaves no trace in the run.
3. **The preflight reports the driver** (`check_driver`) and turns a CUDA init failure into a clean
   FAIL instead of a traceback.
   `mc_check_cuda` (added after gnode065) runs the real `torch.cuda.init()` right after the driver check,
   so a node whose GPU is dead despite a good driver is recorded in `bad_nodes.txt` and skipped the same way.
4. **Resubmission after a bad landing is automatic (2026-09-28, user decision).** Manual resubmission cost
   the first arm 5.5 idle hours: job 2721197 landed on gnode047 (driver 570, never probed) at 17:37 and
   died in its first second, and nobody resubmitted until 23:02. Now `mc_bad_node_exit` (slurm/common.sh)
   records the node and `mc_resubmit_elsewhere` resubmits the same job before exiting 6: the script from
   `scontrol show job`'s `Command=`, the same name and `TimeLimit`, `--export=ALL` (the job's own
   environment: `MC_CONFIG`, `MC_SEED`, `MC_LAYOUT`, `MC_RESUME_PHASE`, ...) and `mc_sbatch_args`, whose
   exclude list now contains the node. Limits: at most `MC_RESUBMIT_MAX` (5) attempts per submission
   (`MC_RESUBMIT_ATTEMPT` counts them), never for a job pinned with `-w` (its checkpoints or checkpoint
   under evaluation live on that node), off with `MC_AUTO_RESUBMIT=0`; `MC_RESUBMIT_DRY_RUN=1` prints the
   command instead (checked on Ada against the queued arm job 2721833: identical flags to `make sbatch-train`).
   Training jobs (`MC_RESUBMIT_SERIAL=1`, set by `train.sbatch`) keep their queue order: a pending
   same-name twin held by `--dependency=singleton` would start before a new submission, and a singleton
   resubmission would wait behind that twin, because singleton also waits for earlier *pending* jobs of
   the same name (observed: twin 2721834 sat at `singleton(unfulfilled)` while 2721833 was only pending).
   So the new job is submitted `--hold`, every pending same-name job is re-pointed to
   `afterany:<new>,singleton`, and the new job is released; if re-pointing fails the new job is cancelled
   rather than risk two live jobs on one run dir. The resume test now runs the node checks before it
   queues phase 2 (2720629 had queued phase 2 first, which then ran on the same dead gnode077).

## Consequences

* Only ~7 verified nodes today (more once the remaining 19 are probed or upgraded); with checkpoints
  pinned to one node (009), a pinned node that is busy means waiting. `-w <node>` from `node.txt` still
  wins over the exclude list because a pinned node has run the wheels before.
* Ask the admins which 2080 Ti nodes are scheduled for the 580 upgrade; the exclude list shrinks as they
  land. To re-probe: `for n in ...; do srun -A research --qos=low -p u22 -w $n -N1 -n1 -c1 --mem=300M
  -t 1 cat /proc/driver/nvidia/version; done` (research/low allows one node per job).
