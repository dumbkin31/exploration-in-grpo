from __future__ import annotations

import math

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
