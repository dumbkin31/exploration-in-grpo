#!/usr/bin/env python
"""Carry a run directory between machines through a private Hugging Face Hub model repo (decision 014).

Kaggle sessions end after 12 hours and lose their disk, so kaggle/train.sh pulls a run at the start of a
session and pushes every new checkpoint. It also moves checkpoints between sites, e.g. to evaluate a
Jarvislabs run on Kaggle.

    python scripts/hub_sync.py push --run math_mixed_cuts-t4-s1      # newest checkpoint + small files
    python scripts/hub_sync.py pull --run math_mixed_cuts-t4-s1      # newest checkpoint + small files
    python scripts/hub_sync.py pull --run math_grpo-h100-s1 --step 100
    python scripts/hub_sync.py status                                 # runs on the Hub, newest step each

Repo: MC_HUB_REPO (created private on the first push). Token: HF_TOKEN, with write access to push.
Layout in the repo mirrors the run dir: <run>/checkpoints/global_step_<N>/..., the tracker
<run>/checkpoints/latest_checkpointed_iteration.txt, <run>/{metrics.jsonl,phases.jsonl,node.txt,
memory_profile.md}, <run>/eval/**, <run>/launcher/*.log. Not carried: rollout_dumps/, cuts_stats/ (the
per-step diagnostics are in metrics.jsonl and W&B), wandb/ (W&B resumes by run id), gpu_mem.jsonl.
Only the newest MC_HUB_KEEP (default 2) checkpoints stay on the Hub; a checkpoint already there is not
uploaded again.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

SMALL_FILES = ("metrics.jsonl", "phases.jsonl", "node.txt", "memory_profile.md")
TRACKER = "checkpoints/latest_checkpointed_iteration.txt"
STEP_RE = re.compile(r"^checkpoints/global_step_(\d+)/")


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def _remote(api: Any, repo: str, run: str) -> list[str]:
    """Paths relative to <run>/ of every file of this run on the Hub ([] if the repo does not exist)."""
    try:
        files = api.list_repo_files(repo)
    except Exception as e:  # noqa: BLE001 - a missing repo is the normal first-push case
        if "404" in str(e) or "not found" in str(e).lower() or type(e).__name__ == "RepositoryNotFoundError":
            return []
        raise
    prefix = f"{run}/"
    return [f[len(prefix) :] for f in files if f.startswith(prefix)]


def _steps(paths: list[str]) -> list[int]:
    return sorted({int(m.group(1)) for p in paths if (m := STEP_RE.match(p))})


def _local_step(run_dir: Path) -> int | None:
    try:
        return int((run_dir / TRACKER).read_text().strip())
    except (OSError, ValueError):
        return None


def local_files_to_push(run_dir: Path, step: int | None) -> list[Path]:
    files: list[Path] = [run_dir / f for f in SMALL_FILES if (run_dir / f).is_file()]
    if (run_dir / "eval").is_dir():
        files += [p for p in sorted((run_dir / "eval").rglob("*")) if p.is_file()]
    if (run_dir / "launcher").is_dir():
        files += sorted((run_dir / "launcher").glob("*.log"))
    if step is not None:
        ck = run_dir / "checkpoints" / f"global_step_{step}"
        files += [p for p in sorted(ck.rglob("*")) if p.is_file()]
        files.append(run_dir / TRACKER)
    return files


def push(api: Any, repo: str, runs_dir: Path, run: str, keep: int = 2) -> dict[str, Any]:
    """Upload the newest checkpoint (unless the Hub has it) and the small files; prune old checkpoints."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    run_dir = runs_dir / run
    if not run_dir.is_dir():
        raise SystemExit(f"no run dir {run_dir}")
    api.create_repo(repo, private=True, exist_ok=True)
    remote = _remote(api, repo, run)
    remote_steps = _steps(remote)
    step = _local_step(run_dir)
    upload_ckpt = step is not None and step not in remote_steps
    files = local_files_to_push(run_dir, step if upload_ckpt else None)
    ops: list[Any] = [
        CommitOperationAdd(path_in_repo=f"{run}/{p.relative_to(run_dir).as_posix()}", path_or_fileobj=str(p))
        for p in files
    ]
    keep_steps = sorted(set(remote_steps) | ({step} if step is not None else set()))[-keep:]
    stale = [s for s in remote_steps if s not in keep_steps]
    ops += [
        CommitOperationDelete(path_in_repo=f"{run}/checkpoints/global_step_{s}/", is_folder=True)
        for s in stale
    ]
    if ops:
        api.create_commit(
            repo, operations=ops, commit_message=f"{run}: step {step} ({len(files)} files, -{len(stale)} old)"
        )
    return {
        "run": run,
        "step": step,
        "uploaded_checkpoint": upload_ckpt,
        "files": len(files),
        "pruned": stale,
    }


def pull(api: Any, repo: str, runs_dir: Path, run: str, step: int | None = None) -> dict[str, Any]:
    """Download a checkpoint (default: the Hub's newest) and the small files into <runs_dir>/<run>/."""
    from huggingface_hub import hf_hub_download

    remote = _remote(api, repo, run)
    if not remote:
        return {"run": run, "step": None, "files": 0}
    steps = _steps(remote)
    if step is None and steps:
        step = steps[-1]
    if step is not None and step not in steps:
        raise SystemExit(f"{run}: step {step} is not on {repo} (have {steps})")
    run_dir = runs_dir / run
    have = _local_step(run_dir)
    wanted = [
        p
        for p in remote
        if p in SMALL_FILES
        or p.startswith(("eval/", "launcher/"))
        or (step is not None and p.startswith(f"checkpoints/global_step_{step}/") and have != step)
    ]
    for p in wanted:
        hf_hub_download(repo, f"{run}/{p}", local_dir=str(runs_dir))
    if step is not None:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / TRACKER).write_text(
            f"{step}\n"
        )  # the checkpoint just pulled, whatever the Hub tracker says
    return {"run": run, "step": step, "files": len(wanted)}


def status(api: Any, repo: str) -> dict[str, list[int]]:
    try:
        files = api.list_repo_files(repo)
    except Exception:  # noqa: BLE001
        return {}
    runs: dict[str, list[int]] = {}
    for f in files:
        run, _, rest = f.partition("/")
        if rest:
            runs.setdefault(run, [])
            if m := STEP_RE.match(rest):
                runs[run].append(int(m.group(1)))
    return {r: sorted(set(s)) for r, s in sorted(runs.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull", "status"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--repo", default=os.environ.get("MC_HUB_REPO"))
    ap.add_argument("--runs-dir", default=os.environ.get("MC_RUNS_DIR"))
    ap.add_argument("--step", type=int, default=None, help="pull: this checkpoint instead of the newest")
    ap.add_argument("--keep", type=int, default=int(os.environ.get("MC_HUB_KEEP", "2")))
    args = ap.parse_args()
    if not args.repo:
        print("ERROR: --repo or MC_HUB_REPO required (e.g. <hf-user>/mixed-cuts-runs)", file=sys.stderr)
        return 2
    api = _api()
    if args.action == "status":
        for run, steps in status(api, args.repo).items():
            print(f"{run}: checkpoints {steps or '-'}")
        return 0
    if not args.run or not args.runs_dir:
        print("ERROR: --run and --runs-dir (or MC_RUNS_DIR) required", file=sys.stderr)
        return 2
    fn = push if args.action == "push" else pull
    kw = {"keep": args.keep} if args.action == "push" else {"step": args.step}
    print(fn(api, args.repo, Path(args.runs_dir), args.run, **kw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
