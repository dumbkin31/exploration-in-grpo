# 017: One reward actor on the Ada 1-GPU layout

Date: 2026-10-26. Status: proposed; not yet measured on Ada. Amends the host budget of
[012](012-research-qos-one-gpu-lora.md).

## Problem

verl starts `reward.num_workers` reward actors (`RewardLoopWorker`; default 8,
`verl/trainer/config/reward/reward.yaml:5`, created in `experimental/reward_loop/reward_loop.py:324-341`).
No config of ours set it, so `research_1gpu` (and `nlp_1gpu`, which uses the same layout) ran 8. Each actor
is a separate Python process with verl and the tokenizer loaded, and each spawns its own math-verify pool of
`MC_REWARD_WORKERS` processes (2 on Ada, `configs/ada.env.sh`; `src/mixed_cuts/reward.py`).

Per-process memory, from the H200 sidecar (`gpu_mem.jsonl`, same software stack):

| Process | RSS | Anonymous |
|---|---|---|
| `ray::RewardLoopWorker` | 742-885 MiB | 543 MiB |
| math-verify worker (`python`, spawned) | ~100 MiB (63 MiB measured on a laptop) | ~85 MiB (median) |

8 x (543 + 2 x 85) MiB is about 5.7 GiB of a 30 GB cgroup. Decision 012 measured that cgroup at 29.4 GB
steady and up to 29,996 MiB at peak, and counted "agent/TQ/reward/dataloader workers ~5 GB".

## Decision

`reward.num_workers: 1` in `configs/train/layout/research_1gpu.yaml`, pinned by
`tests/test_layout_guards.py`. That leaves about 0.7 GiB for rewards, roughly 5 GiB less once every pool has
spawned. `MC_REWARD_WORKERS` stays 2.

## Why one is enough

* Same rewards. The reward is a pure function of the response, so the number of actors does not change any
  score. Both arms use the same layout.
* No Ray placement change. Actors created with default options hold no CPU for their lifetime (Ray 2.56.1
  `python/ray/actor.py:1912-1916`); the reward actors were never in the ~10 logical CPUs of the
  `ray_init.num_cpus` comment.
* Throughput. One 2080 Ti produces about 0.6 responses/s (012: 2048 sequences in a 59-minute rollout).
  Exact matches skip math-verify. Two math-verify workers handle tens of ordinary comparisons per second,
  and a pathological answer holds a worker for about 6 s. Even a burst of 32 of those (`max_num_seqs`)
  clears in under 2 minutes.
* Waiting is safe since PR #23. With one actor, every reward call in the job queues on the same 2 workers.
  Since PR #23, the wait for a free worker no longer counts against the 30 s math-verify timeout. Before it,
  a burst of slow answers would have timed out correct answers queued behind them. This change must not
  land without PR #23.

## Not changed

* `nlp_4gpu`: 40 cores and 117 GB; not memory-bound.
* `h100_1gpu`: 200 GB of host RAM.
* `kaggle_t4`: the same 30 GB profile with 4 vCPUs and the same default of 8 actors. It is the obvious next
  candidate. It is left out here to keep this change to Ada.

## To confirm

The next Ada smoke (`make sbatch-smoke`) should show `jobs/<id>/host_mem_peak.txt` several GB lower than
smoke 2719378's 29,996 MiB, and only one `ray::reward_loop_worker_*` process on the node.
