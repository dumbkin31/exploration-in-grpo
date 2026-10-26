from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mixed_cuts.stability import StabilityAbort, StabilityConfig, StabilityWatch


def _m(grad=1.0, entropy=1.0, **extra):
    return {"actor/grad_norm": grad, "actor/entropy": entropy, "actor/pg_loss": 0.1, **extra}


def test_quiet_run_raises_no_alert():
    w = StabilityWatch(StabilityConfig())
    for step in range(1, 30):
        out = w.check(_m(grad=1.0 + 0.1 * (step % 3), entropy=1.0 - 0.005 * step), step)
        assert out["stability/n_alerts"] == 0


def test_nan_and_inf_alert_and_optional_abort():
    w = StabilityWatch(StabilityConfig())
    out = w.check(_m(**{"actor/pg_loss": float("nan")}), 3)
    assert out["stability/alert_nan"] == 1.0 and out["stability/n_alerts"] == 1
    out = w.check(_m(grad=float("inf")), 4)
    assert out["stability/alert_nan"] == 1.0
    with pytest.raises(StabilityAbort):
        StabilityWatch(StabilityConfig(abort_on_nan=True)).check(_m(**{"actor/kl_loss": math.inf}), 5)


def test_grad_norm_spike_absolute_and_relative():
    w = StabilityWatch(StabilityConfig(grad_norm_max=50.0, grad_norm_spike_factor=10.0))
    for step in range(1, 8):
        assert w.check(_m(grad=1.0), step)["stability/alert_grad_spike"] == 0.0
    out = w.check(_m(grad=15.0), 8)  # 15x the running median of 1.0
    assert out["stability/alert_grad_spike"] == 1.0 and out["stability/grad_norm_over_median"] == 15.0
    assert w.check(_m(grad=60.0), 9)["stability/alert_grad_spike"] == 1.0  # absolute ceiling
    assert StabilityWatch(StabilityConfig()).check(_m(grad=40.0), 1)["stability/alert_grad_spike"] == 0.0


def test_entropy_collapse_only_in_the_window():
    w = StabilityWatch(StabilityConfig(entropy_collapse_window=20, entropy_collapse_ratio=0.25))
    w.check(_m(entropy=2.0), 1)
    assert w.check(_m(entropy=0.6), 2)["stability/alert_entropy_collapse"] == 0.0  # 0.3 x
    out = w.check(_m(entropy=0.4), 3)  # 0.2 x -> collapse
    assert out["stability/alert_entropy_collapse"] == 1.0 and out[
        "stability/entropy_over_step1"
    ] == pytest.approx(0.2)
    assert w.check(_m(entropy=0.1), 25)["stability/alert_entropy_collapse"] == 0.0  # outside the window


def test_rebuild_from_metrics_jsonl_after_resume():
    w = StabilityWatch(StabilityConfig())
    w.rebuild(
        [{"step": s, "actor/grad_norm": 1.0, "actor/entropy": 2.0 if s == 1 else 1.5} for s in range(1, 10)]
    )
    assert (
        w.check(_m(grad=20.0, entropy=0.3), 10)["stability/n_alerts"] == 2
    )  # spike vs median 1.0 + collapse vs 2.0


def test_disabled_and_config_mapping():
    w = StabilityWatch(StabilityConfig.from_mapping({"enabled": False}))
    assert w.check(_m(grad=float("nan")), 1)["stability/alert_nan"] == 0.0
    assert StabilityConfig.from_mapping(None).enabled


# --- policy drift: the onset of the H200 divergence (decisions 016, 018) ---

H200 = Path(__file__).resolve().parent.parent / "results" / "h200-2026-10-01" / "training"
YAML_DRIFT = StabilityConfig(abort_on_divergence=True)  # the other drift fields are base_grpo.yaml's values


def _h200_steps(run: str) -> list[tuple[int, dict]]:
    rows = {}
    for line in (H200 / run / "metrics.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r.get("step"), int):
            rows[r["step"]] = r
    return sorted(rows.items())


@pytest.mark.parametrize(
    "run,first_alert,abort_at",
    [("math_mixed_cuts-h200-s1", 49, 50), ("math_grpo-h200-s1", 66, 67)],
)
def test_the_h200_divergences_are_stopped_at_their_onset(run, first_alert, abort_at):
    w = StabilityWatch(YAML_DRIFT)
    alerts = []
    with pytest.raises(StabilityAbort, match="policy drift for 2 consecutive steps") as stop:
        for step, metrics in _h200_steps(run):
            out = w.check(metrics, step)
            if out["stability/alert_policy_drift"]:
                alerts.append(step)
    assert alerts == [first_alert], "no drift alert before the divergence began"
    assert f"(step {abort_at})" in str(stop.value)


def test_drift_needs_consecutive_steps():
    w = StabilityWatch(StabilityConfig(abort_on_divergence=True))
    for step in range(1, 8):
        w.check(_m(entropy=0.1, **{"actor/ppo_kl": 0.001}), step)
    assert w.check(_m(entropy=0.1, **{"actor/ppo_kl": 0.05}), 8)["stability/drift_streak"] == 1
    assert w.check(_m(entropy=0.1, **{"actor/ppo_kl": 0.001}), 9)["stability/drift_streak"] == 0
    assert w.check(_m(entropy=0.1, **{"actor/ppo_kl": -0.08}), 10)["stability/alert_policy_drift"] == 1.0
    with pytest.raises(StabilityAbort):
        w.check(_m(entropy=0.1, **{"actor/ppo_kl": 0.03}), 11)


def test_entropy_rise_counts_after_the_warm_up_and_alerts_without_abort_by_default():
    w = StabilityWatch(StabilityConfig())
    assert w.check(_m(entropy=0.1), 1)["stability/alert_policy_drift"] == 0.0
    assert w.check(_m(entropy=0.3), 2)["stability/alert_policy_drift"] == 0.0  # inside entropy_rise_after
    for step in range(3, 7):
        w.check(_m(entropy=0.1), step)
    for step in (7, 8, 9):  # 2.5 x the minimum: alerts, but the dataclass default does not abort
        out = w.check(_m(entropy=0.25), step)
        assert out["stability/alert_policy_drift"] == 1.0 and out["stability/drift_streak"] == step - 6


def test_rebuild_restores_the_entropy_minimum():
    w = StabilityWatch(StabilityConfig())
    w.rebuild([{"step": s, "actor/entropy": 0.5 - 0.04 * s} for s in range(1, 11)])  # minimum 0.1
    assert w.check(_m(entropy=0.21), 11)["stability/alert_policy_drift"] == 1.0
