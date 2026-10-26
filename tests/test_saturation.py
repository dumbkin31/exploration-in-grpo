from __future__ import annotations

import json
from pathlib import Path

from saturation.analyze_saturation import analyze_dataset


def test_analyze_dataset(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "analysis"

    chunk1 = eval_dir / "math_train" / "0-2" / "math_train"
    chunk1.mkdir(parents=True)
    with (chunk1 / "samples.jsonl").open("w", encoding="utf-8") as f:
        # Problem 0: all 16 correct
        for s in range(16):
            f.write(
                json.dumps({"problem": 0, "index": "train-0", "sample": s, "correct": 1.0})
                + "\n"
            )
        # Problem 1: 14 correct, index missing
        for s in range(16):
            f.write(
                json.dumps(
                    {
                        "problem": 1,
                        "index": "None",
                        "sample": s,
                        "correct": 1.0 if s < 14 else 0.0,
                    }
                )
                + "\n"
            )

    chunk2 = eval_dir / "math_train" / "2-3" / "math_train"
    chunk2.mkdir(parents=True)
    with (chunk2 / "samples.jsonl").open("w", encoding="utf-8") as f:
        # Problem 2 (local problem 0 in chunk 2): 0 correct, index None
        for s in range(16):
            f.write(
                json.dumps({"problem": 0, "index": None, "sample": s, "correct": 0.0}) + "\n"
            )

    summary = analyze_dataset(eval_dir, "math_train", out_dir)

    assert summary["dataset"] == "math_train"
    assert summary["num_datapoints"] == 3
    assert summary["total_rollouts_per_datapoint"] == [16]
    assert summary["histogram"] == {0: 1, 14: 1, 16: 1}

    metrics = summary["metrics"]
    assert metrics["all_correct_count"] == 1
    assert metrics["all_wrong_count"] == 1
    assert metrics["zero_variance_collapsed_count"] == 2
    assert round(metrics["zero_variance_collapsed_frac"], 4) == 0.6667
    assert metrics["at_least_15_correct_count"] == 1

    per_example_file = out_dir / "math_train_per_example.jsonl"
    assert per_example_file.exists()
    lines = [json.loads(line) for line in per_example_file.read_text().splitlines() if line]
    assert len(lines) == 3

