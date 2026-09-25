#!/usr/bin/env python
"""Verify a kill-and-resubmit cycle (Task C): the resumed job continued from the last checkpoint.

    python scripts/check_resume.py RUN_DIR --killed-at 15 --final-step 30 --save-freq 5

Asserts, on the durable run dir shared by both jobs:
  * metrics.jsonl steps are 1..killed_at (first job) followed by a restart at
    (last checkpoint before the kill) + 1, continuing to final_step;
  * checkpoints/latest_checkpointed_iteration.txt == final_step and that directory exists;
  * the checkpoint the resume started from existed (global_step_<last_ckpt>);
  * exactly one W&B run id across every offline-run-* directory (WANDB_RUN_ID stable);
  * two jobs/<id>/ directories (two submissions).
Exit 0 on success, 1 with a readable list of failures otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--killed-at", type=int, required=True)
    ap.add_argument("--final-step", type=int, required=True)
    ap.add_argument("--save-freq", type=int, required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    fails: list[str] = []

    steps = []
    for line in (run / "metrics.jsonl").read_text().splitlines() if (run / "metrics.jsonl").exists() else []:
        try:
            steps.append(int(json.loads(line)["step"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    last_ckpt = (args.killed_at // args.save_freq) * args.save_freq
    expected_restart = last_ckpt + 1
    print(f"logged steps: {steps}")
    if not steps:
        fails.append("metrics.jsonl has no steps")
    else:
        # the first job's steps 1..killed_at, then a restart
        try:
            restart_idx = next(i for i in range(1, len(steps)) if steps[i] <= steps[i - 1])
        except StopIteration:
            restart_idx = None
        if restart_idx is None:
            fails.append("no restart found in metrics.jsonl (steps never went backwards)")
        else:
            first, second = steps[:restart_idx], steps[restart_idx:]
            if first[-1] < args.killed_at:
                fails.append(f"first job stopped at step {first[-1]} < kill step {args.killed_at}")
            if second[0] != expected_restart:
                fails.append(
                    f"resumed job started at step {second[0]}, expected {expected_restart} (last checkpoint {last_ckpt} + 1)"
                )
            if second[-1] != args.final_step:
                fails.append(f"resumed job ended at step {second[-1]}, expected {args.final_step}")
            print(f"first job: {first[0]}..{first[-1]}; resumed job: {second[0]}..{second[-1]}")

    ck = run / "checkpoints"
    tracker = ck / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        fails.append(f"{tracker} missing")
    else:
        latest = int(tracker.read_text().strip())
        if latest != args.final_step:
            fails.append(f"latest_checkpointed_iteration = {latest}, expected {args.final_step}")
        if not (ck / f"global_step_{latest}").is_dir():
            fails.append(f"checkpoint dir global_step_{latest} missing")
    if not str(ck).startswith("/share1") and not str(ck.resolve()).startswith("/share1"):
        fails.append(f"checkpoints are not under /share1: {ck}")

    ids = set()
    for d in (run / "wandb").glob("offline-run-*"):
        ids.add(d.name.rsplit("-", 1)[-1])
    if len(ids) != 1:
        fails.append(f"expected exactly one W&B run id across offline runs, found {sorted(ids)}")
    jobs = sorted(p.name for p in (run / "jobs").glob("*")) if (run / "jobs").exists() else []
    if len(jobs) < 2:
        fails.append(f"expected 2 job dirs (two submissions), found {jobs}")

    if fails:
        print("RESUME CHECK FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(
        f"RESUME CHECK OK: killed at {args.killed_at}, resumed at {expected_restart}, finished at {args.final_step}, one W&B run id"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
