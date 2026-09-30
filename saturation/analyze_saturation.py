#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_dataset(eval_dir: Path, dataset: str, output_dir: Path) -> dict:
    samples_path = eval_dir / dataset / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {samples_path}")

    groups = defaultdict(list)

    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = record.get("index") or f"problem-{record['problem']}"
            groups[str(key)].append(float(record["correct"]))

    rows = []
    for index, scores in groups.items():
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

    return {
        "dataset": dataset,
        "num_datapoints": len(rows),
        "total_rollouts_per_datapoint": sorted(
            {row["total_rollouts"] for row in rows}
        ),
        "histogram": dict(sorted(histogram.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["math_train", "dapo_train"],
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)

    summaries = {}
    for dataset in args.datasets:
        summaries[dataset] = analyze_dataset(eval_dir, dataset, output_dir)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 6))

    for dataset, summary in summaries.items():
        histogram = {
            int(key): value for key, value in summary["histogram"].items()
        }
        x_values = sorted(histogram)
        y_values = [histogram[x] for x in x_values]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            label=dataset,
        )

    max_rollouts = max(
        (
            max(map(int, summary["histogram"].keys()))
            for summary in summaries.values()
            if summary["histogram"]
        ),
        default=0,
    )

    plt.xticks(range(max_rollouts + 1))
    plt.xlabel("Number of correct rollouts")
    plt.ylabel("Number of data points")
    plt.title("Correct-rollout distribution by dataset")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "saturation_plot.png"
    plt.savefig(plot_path, dpi=180)
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()