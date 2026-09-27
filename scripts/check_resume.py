#!/usr/bin/env python
"""Verify a kill-and-resubmit cycle (Task C): the resumed job continued from the last checkpoint.

    python scripts/check_resume.py RUN_DIR --killed-at 15 --final-step 30 --save-freq 5

Asserts, on the durable run dir shared by both jobs:
  * metrics.jsonl steps are 1..killed_at (first job) followed by a restart at
    (last checkpoint before the kill) + 1, continuing to final_step (the restart is a backwards
    step, or, when the kill fell exactly on a checkpoint, the second job's start time from
    jobs/<id>/job_info.txt);
  * <checkpoints dir>/latest_checkpointed_iteration.txt == final_step and that directory exists
    (pass --checkpoints-dir when checkpoints live on node-local scratch, docs/decisions/009);
  * the checkpoint the resume started from existed (global_step_<last_ckpt>);
  * exactly one W&B run id across every wandb/run-* (online) and offline-run-* directory (WANDB_RUN_ID stable);
  * two jobs/<id>/ directories (two submissions).
Exit 0 on success, 1 with a readable list of failures otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def wandb_run_ids(run: Path) -> set[str]:
    """Run ids of every W&B run folder under <run>/wandb: `run-<ts>-<id>` (online), `offline-run-<ts>-<id>`,
    and the nested `wandb/wandb/...` layout older jobs produced."""
    ids: set[str] = set()
    for pattern in ("wandb/run-*", "wandb/offline-run-*", "wandb/wandb/run-*", "wandb/wandb/offline-run-*"):
        for d in run.glob(pattern):
            if d.is_dir():
                ids.add(d.name.split("-", 2)[-1] if d.name.startswith("run-") else d.name.split("-", 3)[-1])
    return ids


def find_restart(steps: list[int], times: list[float], jobs_dir: Path) -> int | None:
    """Index of the first metrics row logged by the resumed job.

    When the kill lands between two checkpoints the resumed job re-logs some steps, so the step
    sequence goes backwards. When the kill lands exactly on a checkpoint (kill at 15 with save_freq 5,
    the resume test's own setting) the resume is seamless (..., 15, 16, ...) and the restart is only
    visible from the second job's start time (jobs/<second id>/job_info.txt, written by mc_job_init).
    """
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            return i
    infos = sorted(
        (p for p in jobs_dir.glob("*/job_info.txt")),
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 0,
    )
    if len(infos) < 2:
        return None
    second_job = infos[1].parent.name
    # node.txt (mc_check_node_pin) holds one block per submission: node=, run_dir=, job=, date=%F %T
    started = None
    node_txt = jobs_dir.parent / "node.txt"
    if node_txt.exists():
        block_job = None
        for line in node_txt.read_text().splitlines():
            if line.startswith("job="):
                block_job = line[4:].strip()
            elif line.startswith("date=") and block_job == second_job:
                started = datetime.strptime(line[5:].strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    if started is None:  # job_info.txt is written at job init: its mtime is the second start
        started = infos[1].stat().st_mtime
    for i, t in enumerate(times):
        if t and t >= started:
            return i if i > 0 else None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--killed-at", type=int, required=True)
    ap.add_argument("--final-step", type=int, required=True)
    ap.add_argument("--save-freq", type=int, required=True)
    ap.add_argument(
        "--checkpoints-dir",
        default=None,
        help="default: <run_dir>/checkpoints (scratch layouts keep them elsewhere)",
    )
    args = ap.parse_args()
    run = Path(args.run_dir)
    fails: list[str] = []

    steps: list[int] = []
    times: list[float] = []
    for line in (run / "metrics.jsonl").read_text().splitlines() if (run / "metrics.jsonl").exists() else []:
        try:
            row = json.loads(line)
            steps.append(int(row["step"]))
            times.append(float(row.get("time", 0.0)))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    last_ckpt = (args.killed_at // args.save_freq) * args.save_freq
    expected_restart = last_ckpt + 1
    print(f"logged steps: {steps}")
    if not steps:
        fails.append("metrics.jsonl has no steps")
    else:
        restart_idx = find_restart(steps, times, run / "jobs")
        if restart_idx is None:
            fails.append(
                "no restart found: steps never went backwards and jobs/*/job_info.txt gives no second start time"
            )
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

    ck = Path(args.checkpoints_dir) if args.checkpoints_dir else run / "checkpoints"
    tracker = ck / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        fails.append(f"{tracker} missing")
    else:
        latest = int(tracker.read_text().strip())
        if latest != args.final_step:
            fails.append(f"latest_checkpointed_iteration = {latest}, expected {args.final_step}")
        if not (ck / f"global_step_{latest}").is_dir():
            fails.append(f"checkpoint dir global_step_{latest} missing")

    ids = wandb_run_ids(run)
    if len(ids) != 1:
        fails.append(f"expected exactly one W&B run id across the run folders, found {sorted(ids)}")
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
