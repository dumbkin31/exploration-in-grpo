"""End-to-end test of the eval runner with a stub generator (no vLLM, no GPU)."""

from __future__ import annotations

import json

import pandas as pd

from mc_data.schema import DS_MATH500, Row
from mc_eval.runner import markdown_table, run_all


def _stub_generator(conversations, n, sampling_cfg):
    # problem 0: 3/4 correct incl. an equivalent form; problem 1: all wrong, no box in one
    assert n == 4 and sampling_cfg["temperature"] == 0.7
    return [
        ["so \\boxed{\\frac{1}{2}}", "\\boxed{0.5}", "\\boxed{2}", "\\boxed{1/2}"],
        ["\\boxed{9}", "\\boxed{9}", "Answer: 8", "\\boxed{7}"],
    ]


def test_run_all_writes_samples_and_results(tmp_path):
    rows = [
        Row(DS_MATH500, "q1", "\\frac{1}{2}", "test", "a").to_record(),
        Row(DS_MATH500, "q2", "10", "test", "b").to_record(),
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(rows).to_parquet(data_dir / "math500.parquet", index=False)
    cfg = {"n_samples": 4, "k_values": [1, 4], "sampling": {"temperature": 0.7}, "reward": {"timeout": 30}}
    results = run_all(["math500", "aime24"], data_dir, _stub_generator, cfg, tmp_path / "out")
    assert set(results) == {"math500"}  # aime24 parquet missing -> skipped with a warning
    m = results["math500"]["metrics"]
    assert m["n_problems"] == 2 and m["pass@1"] == (3 / 4 + 0) / 2
    assert m["pass@4"] == 0.5 and m["maj@4"] == 0.5 and "pass@1_ci95" in m
    samples = [
        json.loads(line) for line in (tmp_path / "out" / "math500" / "samples.jsonl").read_text().splitlines()
    ]
    assert len(samples) == 8 and sum(s["correct"] for s in samples) == 3
    assert samples[6]["has_boxed"] == 0.0 and samples[6]["pred"] == "8"
    assert (tmp_path / "out" / "summary.json").exists()
    table = markdown_table(results, [1, 4])
    assert "| math500 | 2 |" in table and "pass@4" in table
