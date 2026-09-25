from __future__ import annotations

from mixed_cuts.phases import PhaseLog


def test_phase_log_round_trip(tmp_path):
    log = PhaseLog(tmp_path / "phases.jsonl")
    with log.phase(1, "old_log_prob"):
        pass
    log.begin(1, "rollout")
    log.end(1, "rollout")
    log.end(1, "never_started")  # ignored
    recs = PhaseLog.read(log.path)
    assert [r["phase"] for r in recs] == ["old_log_prob", "rollout"]
    assert all(r["t_end"] >= r["t_start"] and r["step"] == 1 for r in recs)
    assert PhaseLog.read(tmp_path / "missing.jsonl") == []
