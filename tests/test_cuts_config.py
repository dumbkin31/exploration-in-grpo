from __future__ import annotations

import pytest

from cuts.config import EXTRA_ARGS_KEY, CutsParams


def test_round_trip_through_extra_args():
    p = CutsParams(k=7, delta=0.05, t_warm=2, empty_set_fallback="argmax", uid="u", session_id=3, step=9)
    extra = p.to_extra_args()
    assert set(extra) == {EXTRA_ARGS_KEY}
    assert CutsParams.from_extra_args(extra) == p
    assert CutsParams.from_extra_args({EXTRA_ARGS_KEY: p}) is p


def test_standard_requests_parse_to_none():
    assert CutsParams.from_extra_args(None) is None
    assert CutsParams.from_extra_args({}) is None
    assert CutsParams.from_extra_args({"kv_transfer_params": {}}) is None


def test_typos_and_bad_values_fail_loudly():
    with pytest.raises(ValueError, match="unknown CUTS parameter"):
        CutsParams.from_extra_args({EXTRA_ARGS_KEY: {"twarm": 5}})
    with pytest.raises(ValueError):
        CutsParams(k=0)
    with pytest.raises(ValueError):
        CutsParams(delta=1.5)
    with pytest.raises(ValueError):
        CutsParams(t_warm=-1)
    with pytest.raises(ValueError):
        CutsParams(empty_set_fallback="greedy")
    with pytest.raises(TypeError):
        CutsParams.from_extra_args({EXTRA_ARGS_KEY: 5})


def test_paper_defaults():
    p = CutsParams()
    assert (p.k, p.delta, p.t_warm, p.empty_set_fallback) == (5, 0.03, 5, "topk")
    assert p.signature == (5, 0.03, "topk")
