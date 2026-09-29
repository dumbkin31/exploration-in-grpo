#!/usr/bin/env python
"""Training-curve figures for the report: the arms overlaid on shared axes (decision 013/014).

    python scripts/plot_runs.py                                  # the two H100 arms, seed 1
    python scripts/plot_runs.py --runs math_grpo-t4-s1 math_mixed_cuts-t4-s1 --out figures/

Reads <runs-dir>/<run>/metrics.jsonl (on the training machine, after `scripts/hub_sync.py pull`, or a
copy on a laptop). Writes one PNG per metric plus overview.png (all panels) into --out (default
<runs-dir>/comparison/figures). Needs matplotlib (requirements/dev.txt; not in the cluster env).
Curves: the raw per-step value (faint) and a trailing mean over --smooth steps (solid).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# (file name, title, [(metric key, label suffix, line style)])
PANELS = [
    ("reward", "Training reward (rollout accuracy)", [("critic/score/mean", "", "-")]),
    (
        "advantage_collapse",
        "Advantage collapse rate",
        [("mixed_cuts/advantage_collapse_rate", "", "-")],
    ),
    (
        "saturation",
        "Groups with all-correct / all-wrong rewards",
        [
            ("mixed_cuts/all_correct_frac", " all correct", "-"),
            ("mixed_cuts/all_wrong_frac", " all wrong", "--"),
        ],
    ),
    ("entropy", "Policy entropy", [("actor/entropy", "", "-")]),
    ("response_length", "Mean response length (tokens)", [("response_length/mean", "", "-")]),
    (
        "validation",
        "Validation: MATH-500 mean@4 (100 problems)",
        [("val-core/math500/reward/mean@4", "", "-o")],
    ),
]


def read_series(path: Path) -> dict[str, list[tuple[int, float]]]:
    """metric -> [(step, value)], the last value per step (a step re-logged after a resume counts once)."""
    by_step: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row.get("step"), int):
                by_step.setdefault(row["step"], {}).update(row)
    series: dict[str, list[tuple[int, float]]] = {}
    for step in sorted(by_step):
        for k, v in by_step[step].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("step", "time"):
                series.setdefault(k, []).append((step, float(v)))
    return series


def trailing_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot(runs_dir: Path, runs: list[str], out: Path, smooth: int) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib missing ({e}); pip install matplotlib (requirements/dev.txt)") from e
    data = {run: read_series(runs_dir / run / "metrics.jsonl") for run in runs}
    out.mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    written: list[Path] = []

    def draw(ax, keys) -> bool:
        drew = False
        for i, run in enumerate(runs):
            for key, suffix, style in keys:
                pts = data[run].get(key) or []
                if not pts:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                c = colors[i % len(colors)]
                if style == "-o" or len(pts) < 3:  # sparse series (validation): points, no smoothing
                    ax.plot(xs, ys, "-o", color=c, markersize=3, label=f"{run}{suffix}")
                else:
                    ax.plot(xs, ys, style, color=c, alpha=0.25, linewidth=0.8)
                    ax.plot(
                        xs, trailing_mean(ys, smooth), style, color=c, linewidth=1.8, label=f"{run}{suffix}"
                    )
                drew = True
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if drew:
            ax.legend(fontsize=7)
        return drew

    for name, title, keys in PANELS:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        if draw(ax, keys):
            ax.set_title(title)
            path = out / f"{name}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (_, title, keys) in zip(axes.flat, PANELS, strict=True):
        draw(ax, keys)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{' vs '.join(runs)} (trailing mean over {smooth} steps)", fontsize=11)
    fig.tight_layout()
    overview = out / "overview.png"
    fig.savefig(overview, dpi=150)
    plt.close(fig)
    written.append(overview)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tag = os.environ.get("MC_RUN_TAG", "h100")
    ap.add_argument("--runs-dir", default=os.environ.get("MC_RUNS_DIR"))
    ap.add_argument("--runs", nargs="+", default=[f"math_grpo-{tag}-s1", f"math_mixed_cuts-{tag}-s1"])
    ap.add_argument("--out", default=None, help="default: <runs-dir>/comparison/figures")
    ap.add_argument("--smooth", type=int, default=5, help="trailing-mean window in steps")
    args = ap.parse_args()
    if not args.runs_dir:
        print("ERROR: --runs-dir or MC_RUNS_DIR required", file=sys.stderr)
        return 2
    runs_dir = Path(args.runs_dir)
    out = Path(args.out) if args.out else runs_dir / "comparison" / "figures"
    for p in plot(runs_dir, args.runs, out, args.smooth):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
