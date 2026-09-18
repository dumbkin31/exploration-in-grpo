from __future__ import annotations

import json
import os

from cuts.stats import CutsStatsWriter, read_cuts_stats, summarize_cuts_stats


def _rec(**kw):
    base = {
        "stats_dir": None,
        "step": 1,
        "uid": "u",
        "session_id": 0,
        "n_cuts_steps": 10,
        "mean_set_size": 3.0,
        "n_singleton": 2,
        "n_fallback": 1,
    }
    base.update(kw)
    return base


def test_writer_appends_jsonl_per_pid_and_reader_filters_by_step(tmp_path):
    w = CutsStatsWriter()
    w.write(_rec(stats_dir=str(tmp_path), step=1, uid="a"))
    w.write(_rec(stats_dir=str(tmp_path), step=2, uid="b"))
    w.write(_rec(stats_dir=None, uid="dropped"))  # no stats_dir -> discarded
    w.close()
    files = sorted(tmp_path.glob("cuts_stats_*.jsonl"))
    assert len(files) == 1 and files[0].name == f"cuts_stats_pid{os.getpid()}.jsonl"
    lines = files[0].read_text().splitlines()
    assert len(lines) == 2 and "stats_dir" not in json.loads(lines[0])
    assert [r["uid"] for r in read_cuts_stats(tmp_path)] == ["a", "b"]
    assert [r["uid"] for r in read_cuts_stats(tmp_path, step=2)] == ["b"]
    assert read_cuts_stats(tmp_path / "missing") == []


def test_reader_tolerates_partial_trailing_line(tmp_path):
    p = tmp_path / "cuts_stats_pid1.jsonl"
    p.write_text(json.dumps(_rec(uid="ok")) + "\n" + '{"uid": "trunc')
    assert [r["uid"] for r in read_cuts_stats(tmp_path)] == ["ok"]


def test_summary_is_weighted_by_cuts_steps():
    recs = [
        _rec(n_cuts_steps=10, mean_set_size=4.0, n_singleton=0, n_fallback=0),
        _rec(n_cuts_steps=30, mean_set_size=2.0, n_singleton=15, n_fallback=3),
        _rec(n_cuts_steps=0, mean_set_size=None, n_singleton=0, n_fallback=0),  # shorter than T_warm
    ]
    s = summarize_cuts_stats(recs)
    assert s["cuts/n_rollouts"] == 3
    assert s["cuts/frac_rollouts_never_active"] == 1 / 3
    assert s["cuts/cuts_steps_per_rollout"] == 40 / 3
    assert s["cuts/set_size_mean"] == (10 * 4.0 + 30 * 2.0) / 40
    assert s["cuts/frac_steps_singleton"] == 15 / 40
    assert s["cuts/frac_steps_fallback"] == 3 / 40


def test_summary_of_nothing():
    assert summarize_cuts_stats([]) == {"cuts/n_rollouts": 0.0}
