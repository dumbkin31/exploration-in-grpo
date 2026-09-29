#!/usr/bin/env python
"""Build the results tables for the report from the runs' evaluations and training logs (decision 013).

    python scripts/compare_runs.py                                   # the two H100 arms, seed 1, + the base model
    python scripts/compare_runs.py --runs math_grpo-h100-s1 math_mixed_cuts-h100-s1 --wandb

Reads, under $MC_RUNS_DIR:
  <run>/eval/global_step_<N>/summary.json    (jarvis/eval.sh; the newest step is used)
  <run>/metrics.jsonl                         (one row per training step)
  base-qwen3-1.7b-<tag>/eval/base/summary.json (jarvis|kaggle/eval.sh base; optional reference row)
Writes <out>/comparison.md and <out>/comparison.json:
  1. evaluation: pass@1 [95% CI], pass@16, maj@16 per benchmark and model
  2. training: final-window means of reward, advantage collapse rate, entropy, response length
  3. response length in tokens per benchmark (all / correct / incorrect, % truncated)
  4. compute: training and evaluation GPU-hours per model, and the cost with --rate (compute-credit section)
With --wandb, each run's evaluation lands in its W&B run summary (eval/<benchmark>/<metric>) and the
tables are logged to a separate run "comparison-<tag>" in the same project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

EVAL_COLS = ("pass@1", "pass@16", "maj@16")
TRAIN_KEYS = {
    "train reward": "critic/score/mean",
    "advantage collapse rate": "mixed_cuts/advantage_collapse_rate",
    "ACR_100": "mixed_cuts/ACR_100",
    "all-correct groups": "mixed_cuts/all_correct_frac",
    "all-wrong groups": "mixed_cuts/all_wrong_frac",
    "entropy": "actor/entropy",
    "response length": "response_length/mean",
}
VAL_KEY = "val-core/math500/reward/mean@4"
LENGTH_KEYS = (
    "mean_response_tokens",
    "mean_response_tokens_correct",
    "mean_response_tokens_incorrect",
    "frac_truncated",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def latest_eval(run_dir: Path) -> tuple[str | None, dict[str, Any] | None]:
    """(label, summary) of the newest evaluated checkpoint of a run, or of the base model's eval dir."""
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return None, None
    cands = [d for d in eval_dir.iterdir() if (d / "summary.json").exists()]
    if not cands:
        return None, None

    def step(d: Path) -> int:
        tail = d.name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    best = max(cands, key=step)
    return best.name, json.loads((best / "summary.json").read_text())


def train_summary(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    """Final-window means of the training metrics, the last validation score and the GPU time."""
    steps = [r for r in rows if isinstance(r.get("step"), int)]
    out: dict[str, Any] = {"last_step": steps[-1]["step"] if steps else 0}
    tail = steps[-window:]
    for label, key in TRAIN_KEYS.items():
        vals = [r[key] for r in tail if isinstance(r.get(key), (int, float))]
        out[label] = mean(vals) if vals else None
    vals = [(r["step"], r[VAL_KEY]) for r in steps if isinstance(r.get(VAL_KEY), (int, float))]
    out["last validation (MATH-500 mean@4)"] = vals[-1][1] if vals else None
    # a step re-logged after a resume counts once
    per_step = {
        r["step"]: r["timing_s/step"] for r in steps if isinstance(r.get("timing_s/step"), (int, float))
    }
    out["gpu_hours"] = sum(per_step.values()) / 3600 if per_step else None
    out["mean_minutes_per_step"] = mean(per_step.values()) / 60 if per_step else None
    return out


def fmt_pct(m: dict[str, Any], col: str, ci: bool = False) -> str:
    if col not in m:
        return "-"
    s = f"{100 * m[col]:.1f}"
    lo_hi = m.get(f"{col}_ci95")
    if ci and lo_hi:
        s += f" [{100 * lo_hi[0]:.1f}, {100 * lo_hi[1]:.1f}]"
    return s


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build(
    runs_dir: Path, runs: list[str], base: str | None, window: int, rate: float | None = None
) -> tuple[str, dict[str, Any]]:
    models: list[tuple[str, str | None, dict[str, Any] | None]] = []
    if base:
        label, summ = latest_eval(runs_dir / base)
        models.append((base, label, summ))
    for run in runs:
        label, summ = latest_eval(runs_dir / run)
        models.append((run, label, summ))
    benchmarks: list[str] = []
    for _, _, summ in models:
        for b in summ or {}:
            if b not in benchmarks:
                benchmarks.append(b)

    lines = ["# Mixed-CUTS vs GRPO: results", "", "## Evaluation", ""]
    lines.append("pass@1 with bootstrap 95% CI, then pass@16 and maj@16, all in %. 16 samples per problem.")
    lines.append("")
    head = (
        "| model | checkpoint | "
        + " | ".join(f"{b} pass@1 | {b} pass@16 | {b} maj@16" for b in benchmarks)
        + " |"
    )
    lines += [head, "|---|---|" + "---|" * (3 * len(benchmarks))]
    data: dict[str, Any] = {"eval": {}, "train": {}}
    for name, label, summ in models:
        cells = []
        for b in benchmarks:
            m = ((summ or {}).get(b) or {}).get("metrics") or {}
            cells += [fmt_pct(m, "pass@1", ci=True), fmt_pct(m, "pass@16"), fmt_pct(m, "maj@16")]
        lines.append(f"| {name} | {label or 'not evaluated'} | " + " | ".join(cells) + " |")
        data["eval"][name] = {
            "checkpoint": label,
            "benchmarks": {b: (summ or {}).get(b, {}).get("metrics") for b in benchmarks},
            "lengths": {
                b: {k: (summ or {}).get(b, {}).get(k) for k in LENGTH_KEYS}
                for b in benchmarks
                if (summ or {}).get(b)
            },
            "generation_hours": sum(float(v.get("generation_seconds") or 0) for v in (summ or {}).values())
            / 3600,
        }

    lines += ["", "## Response length (tokens)", ""]
    lines.append("Mean generated tokens per sample: all / correct / incorrect, and % cut off at max_tokens.")
    lines.append("")
    lines += ["| model | " + " | ".join(benchmarks) + " |", "|---|" + "---|" * len(benchmarks)]
    for name, _label, _summ in models:
        cells = []
        for b in benchmarks:
            ln = data["eval"][name]["lengths"].get(b) or {}
            if ln.get("mean_response_tokens") is None:
                cells.append("-")
                continue
            trunc = ln.get("frac_truncated")
            cells.append(
                f"{ln['mean_response_tokens']:.0f} / {fmt(ln.get('mean_response_tokens_correct'), 0)} / "
                f"{fmt(ln.get('mean_response_tokens_incorrect'), 0)}"
                + (f" ({100 * trunc:.1f}%)" if trunc is not None else "")
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Training", "", f"Means over the last {window} logged steps.", ""]
    cols = ["last step", *TRAIN_KEYS, "last validation (MATH-500 mean@4)"]
    lines += ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for run in runs:
        t = train_summary(read_jsonl(runs_dir / run / "metrics.jsonl"), window)
        data["train"][run] = t
        lines.append(f"| {run} | {t['last_step']} | " + " | ".join(fmt(t[c]) for c in cols[1:]) + " |")

    lines += ["", "## Compute", ""]
    cost_col = f" | cost at INR {rate:g}/GPU-h" if rate else ""
    lines += [
        f"| model | training GPU-hours | minutes per step | evaluation GPU-hours{cost_col} |",
        "|---|---|---|---|" + ("---|" if rate else ""),
    ]
    total = 0.0
    for name, _label, _summ in models:
        t = data["train"].get(name) or {}
        train_h = t.get("gpu_hours") or 0.0
        eval_h = data["eval"][name]["generation_hours"]
        total += train_h + eval_h
        cost = f" | {rate * (train_h + eval_h):,.0f}" if rate else ""
        lines.append(
            f"| {name} | {fmt(t.get('gpu_hours'), 1)} | {fmt(t.get('mean_minutes_per_step'), 1)} | {eval_h:.1f}{cost} |"
        )
    data["compute"] = {"gpu_hours_total": total, "rate_inr": rate, "cost_inr": rate * total if rate else None}
    lines += [
        "",
        f"Total: {total:.1f} GPU-hours"
        + (f", INR {rate * total:,.0f}" if rate else "")
        + ". Training hours are the",
        "sum of logged step times; evaluation hours are vLLM generation time. Setup, smoke tests, model",
        "loading and idle time come on top: take the billed total from the provider's invoice.",
    ]
    return "\n".join(lines) + "\n", data


def push_wandb(data: dict[str, Any], markdown: str, tag: str) -> None:
    import wandb

    entity, project = os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT", "mixed-cuts")
    for run, ev in data["eval"].items():
        if run not in data["train"]:
            continue  # the base model has no training run
        summary = {}
        for bench, m in (ev.get("benchmarks") or {}).items():
            for col in EVAL_COLS:
                if m and col in m:
                    summary[f"eval/{bench}/{col}"] = m[col]
            tokens = ((ev.get("lengths") or {}).get(bench) or {}).get("mean_response_tokens")
            if tokens is not None:
                summary[f"eval/{bench}/mean_response_tokens"] = tokens
        if summary:
            r = wandb.init(entity=entity, project=project, id=run, resume="allow")
            r.summary.update({**summary, "eval/checkpoint": ev.get("checkpoint")})
            r.finish()
    r = wandb.init(entity=entity, project=project, name=f"comparison-{tag}", job_type="comparison")
    rows = []
    for run, ev in data["eval"].items():
        for bench, m in (ev.get("benchmarks") or {}).items():
            if m:
                rows.append([run, ev.get("checkpoint"), bench, *[m.get(c) for c in EVAL_COLS]])
    r.log({"eval_table": wandb.Table(columns=["model", "checkpoint", "benchmark", *EVAL_COLS], data=rows)})
    r.log({"report": wandb.Html(f"<pre>{markdown}</pre>")})
    r.finish()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tag = os.environ.get("MC_RUN_TAG", "h100")
    ap.add_argument("--runs-dir", default=os.environ.get("MC_RUNS_DIR"))
    ap.add_argument("--runs", nargs="+", default=[f"math_grpo-{tag}-s1", f"math_mixed_cuts-{tag}-s1"])
    ap.add_argument("--base", default=f"base-qwen3-1.7b-{tag}", help="reference eval dir name ('' to skip)")
    ap.add_argument("--window", type=int, default=10, help="final steps averaged in the training table")
    ap.add_argument("--out", default=None, help="default: <runs-dir>/comparison")
    ap.add_argument(
        "--wandb", action="store_true", help="write eval metrics into the W&B runs + a comparison run"
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=float(os.environ["MC_GPU_RATE_INR"]) if os.environ.get("MC_GPU_RATE_INR") else None,
        help="INR per GPU-hour for the cost column (e.g. 112.59: H100 spot on Jarvislabs); default MC_GPU_RATE_INR",
    )
    args = ap.parse_args()
    if not args.runs_dir:
        print("ERROR: --runs-dir or MC_RUNS_DIR required (source configs/ada.env.sh)", file=sys.stderr)
        return 2
    runs_dir = Path(args.runs_dir)
    markdown, data = build(runs_dir, args.runs, args.base or None, args.window, args.rate)
    out = Path(args.out) if args.out else runs_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(markdown)
    (out / "comparison.json").write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    print(markdown)
    print(f"written: {out / 'comparison.md'}, {out / 'comparison.json'}")
    if args.wandb:
        push_wandb(data, markdown, tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
