"""compare_runs builds the report tables from eval summaries and metrics.jsonl (decision 013)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("compare_runs", REPO / "scripts" / "compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _summary(p1: float) -> dict:
    m = {"pass@1": p1, "pass@1_ci95": [p1 - 0.02, p1 + 0.02], "pass@16": 0.9, "maj@16": p1 + 0.05}
    lengths = {
        "mean_response_tokens": 700.0,
        "mean_response_tokens_correct": 600.0,
        "mean_response_tokens_incorrect": 1200.0,
        "frac_truncated": 0.01,
        "generation_seconds": 1800.0,
    }
    return {
        "math500": {"metrics": m, "n_problems": 500, **lengths},
        "aime24": {"metrics": dict(m, **{"pass@1": p1 / 4}), "n_problems": 30, "generation_seconds": 1800.0},
    }


def _run(root: Path, name: str, p1: float, acr: float, steps: int = 12) -> None:
    run = root / name
    for step, p in ((50, p1 - 0.1), (100, p1)):
        d = run / "eval" / f"global_step_{step}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(_summary(p)))
    rows = [
        {
            "step": s,
            "critic/score/mean": 0.7,
            "mixed_cuts/advantage_collapse_rate": acr,
            "timing_s/step": 360.0,
        }
        for s in range(1, steps + 1)
    ]
    rows.append({"step": steps, "critic/score/mean": 0.7, "timing_s/step": 360.0})  # re-logged after a resume
    rows[4]["val-core/math500/reward/mean@4"] = 0.71
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_tables_use_the_newest_eval_and_final_window(tmp_path: Path):
    mod = _load()
    _run(tmp_path, "math_grpo-h100-s1", 0.72, acr=0.40)
    _run(tmp_path, "math_mixed_cuts-h100-s1", 0.75, acr=0.20)
    base = tmp_path / "base-qwen3-1.7b-h100" / "eval" / "base"
    base.mkdir(parents=True)
    (base / "summary.json").write_text(json.dumps(_summary(0.70)))
    md, data = mod.build(
        tmp_path, ["math_grpo-h100-s1", "math_mixed_cuts-h100-s1"], "base-qwen3-1.7b-h100", 10
    )
    assert data["eval"]["math_mixed_cuts-h100-s1"]["checkpoint"] == "global_step_100"
    assert "| math_mixed_cuts-h100-s1 | global_step_100 | 75.0 [73.0, 77.0] | 90.0 | 80.0 |" in md
    assert "| base-qwen3-1.7b-h100 | base | 70.0" in md
    t = data["train"]["math_grpo-h100-s1"]
    assert t["last_step"] == 12 and abs(t["advantage collapse rate"] - 0.40) < 1e-9
    assert t["last validation (MATH-500 mean@4)"] == 0.71
    assert abs(t["gpu_hours"] - 12 * 360 / 3600) < 1e-9, "a re-logged step counts once"
    assert abs(t["mean_minutes_per_step"] - 6.0) < 1e-9


def test_missing_eval_and_metrics_render_as_dashes(tmp_path: Path):
    mod = _load()
    (tmp_path / "math_grpo-h100-s1").mkdir()
    md, data = mod.build(tmp_path, ["math_grpo-h100-s1"], None, 10)
    assert "| math_grpo-h100-s1 | not evaluated |" in md
    assert data["train"]["math_grpo-h100-s1"]["gpu_hours"] is None


def test_lengths_and_compute_with_a_rate(tmp_path: Path):
    mod = _load()
    _run(tmp_path, "math_grpo-h100-s1", 0.72, acr=0.40)
    md, data = mod.build(tmp_path, ["math_grpo-h100-s1"], None, 10, rate=112.59)
    assert "| math_grpo-h100-s1 | 700 / 600 / 1200 (1.0%) | - |" in md, "aime24 has no token stats: '-'"
    ev = data["eval"]["math_grpo-h100-s1"]
    assert abs(ev["generation_hours"] - 1.0) < 1e-9, "two benchmarks x 1800 s"
    total = 12 * 360 / 3600 + 1.0
    assert abs(data["compute"]["gpu_hours_total"] - total) < 1e-9
    assert abs(data["compute"]["cost_inr"] - 112.59 * total) < 1e-6
    assert "cost at INR 112.59/GPU-h" in md
