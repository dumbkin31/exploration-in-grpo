"""plot_runs draws the training curves of both arms from metrics.jsonl (report figures)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("plot_runs", REPO / "scripts" / "plot_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _metrics(run: Path, acr: float) -> None:
    run.mkdir(parents=True)
    rows = []
    for s in range(1, 21):
        r = {
            "step": s,
            "critic/score/mean": 0.6 + s / 100,
            "mixed_cuts/advantage_collapse_rate": acr,
            "mixed_cuts/all_correct_frac": acr / 2,
            "mixed_cuts/all_wrong_frac": 0.05,
            "actor/entropy": 0.4,
            "response_length/mean": 800.0,
        }
        if s % 10 == 0:
            r["val-core/math500/reward/mean@4"] = 0.7
        rows.append(r)
    rows.append({"step": 20, "critic/score/mean": 0.9})  # re-logged after a resume: the last value wins
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_series_keep_the_last_value_per_step(tmp_path: Path):
    mod = _load()
    _metrics(tmp_path / "a", 0.4)
    s = mod.read_series(tmp_path / "a" / "metrics.jsonl")
    assert len(s["critic/score/mean"]) == 20 and s["critic/score/mean"][-1] == (20, 0.9)
    assert [p[0] for p in s["val-core/math500/reward/mean@4"]] == [10, 20]
    assert mod.trailing_mean([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 1.5, 2.5, 3.5]


def test_figures_are_written(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load()
    _metrics(tmp_path / "math_grpo-h100-s1", 0.4)
    _metrics(tmp_path / "math_mixed_cuts-h100-s1", 0.2)
    paths = mod.plot(tmp_path, ["math_grpo-h100-s1", "math_mixed_cuts-h100-s1"], tmp_path / "fig", smooth=5)
    names = {p.name for p in paths}
    assert {"reward.png", "advantage_collapse.png", "validation.png", "overview.png"} <= names
    assert all(p.stat().st_size > 5000 for p in paths)
