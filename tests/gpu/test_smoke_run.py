"""Assertions over a finished smoke run (Task B / D5 / D6 / D1 / Task D), given its run dir:

    MC_SMOKE_RUN_DIR=/share1/.../runs/<user>/smoke-s42-<job> make gpu-test

The smoke config runs 2 steps: step 1's |S_t| statistics are summarised at step 2 (one-step lag).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

RUN_DIR = os.environ.get("MC_SMOKE_RUN_DIR")
pytestmark = pytest.mark.skipif(not RUN_DIR or not Path(RUN_DIR).is_dir(), reason="MC_SMOKE_RUN_DIR not set")

REQUIRED_KEYS = [
    "mixed_cuts/advantage_collapse_rate",
    "mixed_cuts/all_correct_frac",
    "mixed_cuts/all_wrong_frac",
    "mixed_cuts/ACR_100",
    "mixed_cuts/rollouts_generated_per_step",
    "mixed_cuts/sample_utilization",
    "mixed_cuts/cost_multiplier",
    "reward/mean",
    "reward/var",
    "reward/group_entropy",
    "reward/std/mean",
    "reward/cuts/mean",
    "response_length/std/mean",
    "response_length/cuts/mean",
    "mixed_cuts/decomp/mu_std",
    "mixed_cuts/decomp/mu_cuts",
    "mixed_cuts/decomp/var_std",
    "mixed_cuts/decomp/var_cuts",
    "mixed_cuts/decomp/between",
    "mixed_cuts/decomp/sigma2_mixed",
    "mixed_cuts/decomp/identity_residual",
    "cuts/frac_old_logprob_uniform_like_k2plus",
    "actor/entropy",
    "actor/grad_norm",
    "response_length/mean",
    "stability/alert_nan",
    "stability/alert_grad_spike",
    "stability/alert_entropy_collapse",
]


def _records():
    path = Path(RUN_DIR) / "metrics.jsonl"
    assert path.exists(), f"{path} missing"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_metrics_have_every_diagnostic_key():
    recs = _records()
    assert len(recs) >= 2
    missing = [k for k in REQUIRED_KEYS if k not in recs[-1]]
    assert not missing, f"missing metrics: {missing}"
    assert recs[-1]["mixed_cuts/cost_multiplier"] == pytest.approx(1.0)
    assert recs[-1]["mixed_cuts/decomp/identity_residual"] < 1e-6


def test_cuts_statistics_arrived_with_one_step_lag():
    recs = [r for r in _records() if r.get("cuts/n_rollouts", 0) > 0]
    assert recs, "no step summarised any CUTS rollout (stats channel broken?)"
    last = recs[-1]
    assert 1.0 < last["cuts/set_size_mean"] < 5.0, f"mean |S_t| = {last['cuts/set_size_mean']}"
    assert last["cuts/stats_step"] == last["step"] - 1
    assert abs(sum(v for k, v in last.items() if k.startswith("cuts/set_size_hist/")) - 1.0) < 1e-6


def test_d1_old_log_probs_are_recomputed_not_engine_logprobs():
    recs = _records()
    fracs = [
        r["cuts/frac_old_logprob_uniform_like_k2plus"]
        for r in recs
        if "cuts/frac_old_logprob_uniform_like_k2plus" in r
    ]
    assert fracs and max(fracs) < 0.5, f"old_log_probs look like log(1/|S_t|): {fracs}"


def test_no_stability_alerts_and_no_think_tags():
    for r in _records():
        assert r["stability/alert_nan"] == 0 and math.isfinite(r["actor/grad_norm"])
        assert r.get("mixed_cuts/frac_responses_with_think_tag", 0.0) == 0.0


def test_profiling_artifacts_exist():
    assert (Path(RUN_DIR) / "phases.jsonl").exists()
    assert (Path(RUN_DIR) / "gpu_mem.jsonl").exists()
    assert (Path(RUN_DIR) / "memory_profile.md").exists()
    assert (Path(RUN_DIR) / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
