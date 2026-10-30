#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sample_paths(eval_dir: Path, dataset: str) -> list[Path]:
    dataset_dir = eval_dir / dataset
    paths = []
    direct = dataset_dir / "samples.jsonl"
    if direct.exists():
        paths.append(direct)
    paths.extend(path for path in sorted(dataset_dir.rglob("samples.jsonl")) if path != direct)
    if not paths:
        raise FileNotFoundError(f"Missing evaluation output under {dataset_dir}")
    return paths


def analyze_dataset(eval_dir: Path, dataset: str, output_dir: Path) -> dict:
    groups = defaultdict(dict)

    for samples_path in _sample_paths(eval_dir, dataset):
        chunk_tag = (
            samples_path.parent.parent.name
            if samples_path.parent.name == dataset
            else samples_path.parent.name
        )
        with samples_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                raw_key = record.get("index")
                if (
                    raw_key is not None
                    and str(raw_key).strip()
                    and str(raw_key).strip().lower() != "none"
                ):
                    key = str(raw_key).strip()
                else:
                    key = f"{chunk_tag}-problem-{record.get('problem', 'unknown')}"

                sample_idx = record.get("sample", len(groups[key]))
                groups[key][sample_idx] = float(record["correct"])

    rows = []
    for index, sample_dict in groups.items():
        scores = list(sample_dict.values())
        num_correct = int(round(sum(scores)))
        total_rollouts = len(scores)

        rows.append(
            {
                "index": index,
                "num_correct": num_correct,
                "total_rollouts": total_rollouts,
                "correct_fraction": num_correct / total_rollouts if total_rollouts else 0.0,
            }
        )

    rows.sort(key=lambda row: row["index"])

    output_dir.mkdir(parents=True, exist_ok=True)

    per_example_path = output_dir / f"{dataset}_per_example.jsonl"
    with per_example_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    histogram = defaultdict(int)
    for row in rows:
        histogram[row["num_correct"]] += 1

    total_prompts = len(rows)
    all_correct_count = 0
    all_wrong_count = 0
    at_least_15_count = 0
    total_rollouts_set = sorted({row["total_rollouts"] for row in rows})

    for row in rows:
        if row["num_correct"] == row["total_rollouts"] and row["total_rollouts"] > 0:
            all_correct_count += 1
        if row["num_correct"] == 0:
            all_wrong_count += 1
        if row["num_correct"] >= 15:
            at_least_15_count += 1

    all_correct_frac = all_correct_count / total_prompts if total_prompts else 0.0
    all_wrong_frac = all_wrong_count / total_prompts if total_prompts else 0.0
    collapsed_frac = (
        (all_correct_count + all_wrong_count) / total_prompts if total_prompts else 0.0
    )
    at_least_15_frac = at_least_15_count / total_prompts if total_prompts else 0.0
    mean_accuracy = (
        sum(r["correct_fraction"] for r in rows) / total_prompts if total_prompts else 0.0
    )

    return {
        "dataset": dataset,
        "num_datapoints": total_prompts,
        "total_rollouts_per_datapoint": total_rollouts_set,
        "histogram": dict(sorted(histogram.items())),
        "metrics": {
            "all_correct_count": all_correct_count,
            "all_correct_frac": round(all_correct_frac, 4),
            "all_wrong_count": all_wrong_count,
            "all_wrong_frac": round(all_wrong_frac, 4),
            "zero_variance_collapsed_count": all_correct_count + all_wrong_count,
            "zero_variance_collapsed_frac": round(collapsed_frac, 4),
            "at_least_15_correct_count": at_least_15_count,
            "at_least_15_correct_frac": round(at_least_15_frac, 4),
            "mean_accuracy": round(mean_accuracy, 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["math_train"],
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)

    summaries = {}
    for dataset in args.datasets:
        summaries[dataset] = analyze_dataset(eval_dir, dataset, output_dir)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    all_rollouts = []
    for summary in summaries.values():
        all_rollouts.extend(summary.get("total_rollouts_per_datapoint", []))
        all_rollouts.extend(map(int, summary["histogram"].keys()))
    max_rollouts = max(all_rollouts, default=16)

    x_values = list(range(max_rollouts + 1))

    # 1. Raw count plot (matches Figure 2 in mid-term report)
    plt.figure(figsize=(10, 6))
    for dataset, summary in summaries.items():
        histogram = {int(k): v for k, v in summary["histogram"].items()}
        y_values = [histogram.get(x, 0) for x in x_values]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            label=dataset,
        )

    plt.xticks(x_values)
    plt.xlabel("Number of correct rollouts")
    plt.ylabel("Number of data points")
    plt.title("Correct-rollout distribution by dataset")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "saturation_plot.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    # 2. Normalized percentage plot (addresses sample-size disparity noted in report)
    plt.figure(figsize=(10, 6))
    for dataset, summary in summaries.items():
        histogram = {int(k): v for k, v in summary["histogram"].items()}
        n_pts = max(1, summary["num_datapoints"])
        y_pct = [(histogram.get(x, 0) / n_pts) * 100 for x in x_values]

        plt.plot(
            x_values,
            y_pct,
            marker="s",
            linewidth=2,
            label=f"{dataset} (N={n_pts})",
        )

    plt.xticks(x_values)
    plt.xlabel("Number of correct rollouts")
    plt.ylabel("Percentage of prompts (%)")
    plt.title("Normalized Correct-rollout distribution (Percentage of Prompts)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    pct_plot_path = output_dir / "saturation_plot_percentage.png"
    plt.savefig(pct_plot_path, dpi=180)
    plt.close()

    print(f"Wrote {plot_path}")
    print(f"Wrote {pct_plot_path}")
    print(f"Wrote {summary_path}")

    print("\n=== Saturation Analysis Summary ===")
    print(
        f"{'Dataset':<15} | {'Prompts':<8} | {'All-Correct':<18} | {'All-Wrong':<18} | {'Collapsed':<18} | {'Pass@1'}"
    )
    print("-" * 90)
    for dataset, summary in summaries.items():
        m = summary["metrics"]
        all_c = f"{m['all_correct_count']} ({m['all_correct_frac']*100:.1f}%)"
        all_w = f"{m['all_wrong_count']} ({m['all_wrong_frac']*100:.1f}%)"
        col = f"{m['zero_variance_collapsed_count']} ({m['zero_variance_collapsed_frac']*100:.1f}%)"
        acc = f"{m['mean_accuracy']*100:.1f}%"
        print(
            f"{dataset:<15} | {summary['num_datapoints']:<8} | {all_c:<18} | {all_w:<18} | {col:<18} | {acc}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
