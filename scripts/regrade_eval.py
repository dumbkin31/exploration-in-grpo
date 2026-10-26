#!/usr/bin/env python
"""Re-grade saved evaluation samples with the current grader, without re-running any model.

    python scripts/regrade_eval.py <root> --out results/h200-2026-10-01/comparison [--workers 8]

<root> holds samples.jsonl files in either layout:
  <run>/eval/<checkpoint>/<benchmark>/samples.jsonl   (laptop backup jarvis-results-2026-10-01/)
  eval/<run>/<checkpoint>/<benchmark>/samples.jsonl   (Hugging Face repo lokola13/mixed-cuts-h200-runs)
Hidden directories (e.g. .hf_upload) are skipped.

Each samples.jsonl keeps every response's full text and ground truth; the sibling results.json holds
the metrics reported by the grader of the time ("old"). Every response is scored again with
mc_eval.runner.grade_problem (the training reward, also for the maj@k answer) and aggregated the same
way ("new"). Writes <out>/regrade.md and <out>/regrade.json: old vs new pass@1, maj@16 and pass@16
per model, checkpoint and benchmark, how many responses flipped each way, and examples of flips.

--write-results DIR also writes re-graded copies of the eval records,
DIR/<run>/eval/<checkpoint>/{summary.json,<benchmark>/results.json}: every metric and the
correct/incorrect token means recomputed, the reported numbers kept under "regraded". That is the
layout scripts/compare_runs.py reads (--runs-dir DIR; training tables need <run>/metrics.jsonl).

--src grades with another source tree (e.g. an older checkout's src/) instead of this repo's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
METRICS = ("pass@1", "maj@16", "pass@16")
MAX_EXAMPLES = 20


def find_samples(root: Path) -> list[tuple[str, str, str, Path]]:
    """(run, checkpoint, benchmark, path) for every samples.jsonl under root, in either layout."""
    found = []
    for path in sorted(root.rglob("samples.jsonl")):
        parts = path.relative_to(root).parts
        if any(p.startswith(".") for p in parts) or len(parts) < 5:
            continue
        if parts[-4] == "eval":  # <run>/eval/<ckpt>/<bench>/samples.jsonl
            run = parts[-5]
        elif parts[-5] == "eval":  # eval/<run>/<ckpt>/<bench>/samples.jsonl
            run = parts[-4]
        else:
            print(f"skipping {path}: unknown layout", file=sys.stderr)
            continue
        found.append((run, parts[-3], parts[-2], path))
    return found


def load_problems(path: Path) -> list[dict[str, Any]]:
    """Group samples.jsonl lines by problem, keeping sample order."""
    problems: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        p = problems.setdefault(
            int(r["problem"]), {"index": r["index"], "gt": r["ground_truth"], "samples": []}
        )
        p["samples"].append(r)
    for p in problems.values():
        p["samples"].sort(key=lambda r: int(r["sample"]))
    return [problems[i] for i in sorted(problems)]


def _mean(xs: list[int]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def regrade_file(
    run: str, ckpt: str, bench: str, path: Path, workers: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(summary row for regrade.md/json, re-graded copy of the benchmark's results.json)."""
    from mc_eval.runner import grade_problem
    from mc_eval.scoring import aggregate

    old = json.loads((path.parent / "results.json").read_text())
    cfg = old["config"]
    problems = load_problems(path)
    n = len(problems[0]["samples"])
    k_values = [int(k) for k in cfg.get("k_values", [1, n]) if int(k) <= n]
    timeout = float(cfg.get("reward", {}).get("timeout", 30))

    def grade(p: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        return grade_problem(bench, p["gt"], [s["text"] for s in p["samples"]], k_values, timeout=timeout)

    # one thread per reward worker: a call never queues behind another, so the timeout means the same
    # as in the sequential eval harness
    with ThreadPoolExecutor(max_workers=workers) as pool:
        graded = list(pool.map(grade, problems))
    new = aggregate([summary for _, summary in graded], seed=int(cfg.get("bootstrap_seed", 0)))

    flips = {"wrong_to_right": 0, "right_to_wrong": 0}
    flipped: Counter[tuple[str, str, str, str]] = Counter()
    tok_correct: list[int] = []
    tok_wrong: list[int] = []
    for p, (rewards, _) in zip(problems, graded, strict=True):
        for s, r in zip(p["samples"], rewards, strict=True):
            if s.get("n_tokens") is not None:
                (tok_correct if r["score"] == 1.0 else tok_wrong).append(int(s["n_tokens"]))
            if float(s["correct"]) == float(r["score"]):
                continue
            key = "wrong_to_right" if r["score"] == 1.0 else "right_to_wrong"
            flips[key] += 1
            flipped[(key, p["index"], p["gt"], r["pred"])] += 1
    examples = [
        {"flip": key, "index": index, "ground_truth": gt, "pred": pred, "n_responses": count}
        for (key, index, gt, pred), count in flipped.most_common(MAX_EXAMPLES)
    ]
    # samples.jsonl and results.json must describe the same eval: pass@1 from the stored 0/1 column
    stored_pass1 = sum(sum(float(s["correct"]) for s in p["samples"]) / n for p in problems) / len(problems)
    regraded = {**old, "metrics": new}
    if tok_correct or tok_wrong:
        regraded["mean_response_tokens_correct"] = _mean(tok_correct)
        regraded["mean_response_tokens_incorrect"] = _mean(tok_wrong)
    regraded["regraded"] = {
        "by": "scripts/regrade_eval.py",
        "metrics_as_evaluated": old["metrics"],
        "mean_response_tokens_correct_as_evaluated": old.get("mean_response_tokens_correct"),
        "mean_response_tokens_incorrect_as_evaluated": old.get("mean_response_tokens_incorrect"),
        **flips,
    }
    row = {
        "model": run,
        "checkpoint": ckpt,
        "benchmark": bench,
        "n_problems": len(problems),
        "n_samples": n,
        "n_responses": len(problems) * n,
        "old": {m: old["metrics"].get(m) for m in METRICS},
        "new": {m: new.get(m) for m in METRICS},
        "new_ci95": {m: new.get(f"{m}_ci95") for m in METRICS},
        **flips,
        "stored_column_matches_results_json": abs(stored_pass1 - old["metrics"]["pass@1"]) < 1e-9,
        "examples": examples,
    }
    return row, regraded


def markdown(rows: list[dict[str, Any]], source: str) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.1f}"

    def cell(r: dict[str, Any], m: str) -> str:
        o, n = r["old"][m], r["new"][m]
        same = o is not None and n is not None and abs(o - n) < 5e-4
        return pct(o) if same else f"{pct(o)} → **{pct(n)}**"

    lines = [
        f"Re-grade of `{source}` with `scripts/regrade_eval.py` (no model re-run). Cells show the published",
        "number, or `old → **new**` where the current grader changes it (in %). Flips count single responses.",
        "",
        "| model | checkpoint | benchmark | pass@1 | maj@16 | pass@16 | wrong→right | right→wrong |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['checkpoint']} | {r['benchmark']} | {cell(r, 'pass@1')} | {cell(r, 'maj@16')}"
            f" | {cell(r, 'pass@16')} | {r['wrong_to_right']} | {r['right_to_wrong']} |"
        )
    total = sum(r["n_responses"] for r in rows)
    w2r, r2w = sum(r["wrong_to_right"] for r in rows), sum(r["right_to_wrong"] for r in rows)
    lines += ["", f"{total} responses re-graded: {w2r} wrong→right, {r2w} right→wrong."]
    bad = [
        f"{r['model']}/{r['checkpoint']}/{r['benchmark']}"
        for r in rows
        if not r["stored_column_matches_results_json"]
    ]
    if bad:
        lines.append(f"WARNING: samples.jsonl does not reproduce results.json pass@1 for: {', '.join(bad)}")
    examples = [(r, e) for r in rows for e in r["examples"]]
    if examples:
        lines += [
            "",
            f"Flipped answers (the {MAX_EXAMPLES} most frequent per model, checkpoint and benchmark):",
            "",
            "| model | checkpoint | benchmark | problem | gold | answer | flip | responses |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r, e in examples:
            lines.append(
                f"| {r['model']} | {r['checkpoint']} | {r['benchmark']} | {e['index']} | `{e['ground_truth']}`"
                f" | `{e['pred']}` | {e['flip']} | {e['n_responses']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory holding the eval samples.jsonl files")
    ap.add_argument("--out", type=Path, required=True, help="writes regrade.md and regrade.json here")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="math-verify subprocesses")
    ap.add_argument("--src", type=Path, default=REPO / "src", help="source tree whose grader is used")
    ap.add_argument("--write-results", type=Path, default=None, help="also write re-graded eval records here")
    args = ap.parse_args()

    os.environ["MC_REWARD_WORKERS"] = str(args.workers)  # read when the reward's pool is first created
    sys.path.insert(0, str(args.src.resolve()))
    from mixed_cuts import reward

    files = find_samples(args.root)
    if not files:
        print(f"ERROR: no samples.jsonl under {args.root}", file=sys.stderr)
        return 2
    rows = []
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for run, ckpt, bench, path in files:
            r, regraded = regrade_file(run, ckpt, bench, path, args.workers)
            rows.append(r)
            summaries.setdefault((run, ckpt), {})[bench] = regraded
            print(
                f"{run} {ckpt} {bench}: pass@1 {r['old']['pass@1']:.4f} -> {r['new']['pass@1']:.4f}, "
                f"flips +{r['wrong_to_right']} -{r['right_to_wrong']}",
                flush=True,
            )
    finally:
        reward.shutdown_pool()
    if args.write_results:
        for (run, ckpt), summary in summaries.items():
            ckpt_dir = args.write_results / run / "eval" / ckpt
            for bench, regraded in summary.items():
                (ckpt_dir / bench).mkdir(parents=True, exist_ok=True)
                (ckpt_dir / bench / "results.json").write_text(
                    json.dumps(regraded, indent=2, sort_keys=True, default=str)
                )
            # the same file eval/run_eval.py writes: {benchmark: results}
            (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "regrade.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    (args.out / "regrade.md").write_text(markdown(rows, args.root.resolve().name))
    print(markdown(rows, args.root.resolve().name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
