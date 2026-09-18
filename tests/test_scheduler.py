"""Tests for the mixed-group planner (pure) and its equivalence to verl's own behaviour."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from cuts.config import EXTRA_ARGS_KEY, CutsParams
from mixed_cuts.scheduler import CUTS, STD, MixedCutsConfig, SessionSpec, plan_group

BASE = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}


def test_n_cuts_zero_reproduces_vanilla_grpo_exactly():
    cfg = MixedCutsConfig(enabled=True, n_std=4, n_cuts=0)
    specs = plan_group(BASE, 4, cfg, uid="u", step=3)
    # verl's _run_prompt: `run_sampling_params = dict(sampling_params)` for every session i in range(n)
    expected = [SessionSpec(i, STD, dict(BASE)) for i in range(4)]
    assert specs == expected
    assert all("extra_args" not in s.sampling_params for s in specs)
    assert all(s.sampling_params is not BASE for s in specs), "copies, never the caller's dict"


def test_disabled_config_is_all_standard_whatever_n():
    cfg = MixedCutsConfig(enabled=False, n_std=8, n_cuts=8)
    specs = plan_group(BASE, 5, cfg, uid="u", step=0)  # n need not match when disabled
    assert [s.rollout_kind for s in specs] == [STD] * 5


def test_validation_and_greedy_rollouts_are_always_standard():
    cfg = MixedCutsConfig(enabled=True, n_std=1, n_cuts=1)
    specs = plan_group(BASE, 2, cfg, uid="u", step=0, force_standard=True)
    assert [s.rollout_kind for s in specs] == [STD, STD]


def test_split_sizes_session_ids_and_extra_args():
    cuts = CutsParams(k=5, delta=0.03, t_warm=5, stats_dir="/tmp/stats")
    cfg = MixedCutsConfig(enabled=True, n_std=3, n_cuts=2, cuts=cuts)
    specs = plan_group(BASE, 5, cfg, uid="prompt-7", step=12)
    assert [s.session_id for s in specs] == [0, 1, 2, 3, 4]
    assert [s.rollout_kind for s in specs] == [STD, STD, STD, CUTS, CUTS]
    for s in specs[:3]:
        assert s.sampling_params == BASE
    for s in specs[3:]:
        sp = s.sampling_params
        assert {k: v for k, v in sp.items() if k != "extra_args"} == BASE
        parsed = CutsParams.from_extra_args(sp["extra_args"])
        assert parsed is not None
        assert (parsed.k, parsed.delta, parsed.t_warm, parsed.stats_dir) == (5, 0.03, 5, "/tmp/stats")
        assert parsed.uid == "prompt-7" and parsed.session_id == s.session_id and parsed.step == 12


def test_existing_extra_args_are_preserved():
    base = {**BASE, "extra_args": {"kv_transfer_params": {"x": 1}}}
    cfg = MixedCutsConfig(enabled=True, n_std=0, n_cuts=1)
    (spec,) = plan_group(base, 1, cfg, uid="u", step=0)
    assert spec.sampling_params["extra_args"]["kv_transfer_params"] == {"x": 1}
    assert EXTRA_ARGS_KEY in spec.sampling_params["extra_args"]
    assert base["extra_args"] == {"kv_transfer_params": {"x": 1}}, "input must not be mutated"


def test_group_size_mismatch_fails_loudly():
    cfg = MixedCutsConfig(enabled=True, n_std=8, n_cuts=8)
    with pytest.raises(ValueError, match="n_std \\+ n_cuts"):
        plan_group(BASE, 4, cfg, uid="u", step=0)
    with pytest.raises(ValueError, match="rollout.n"):
        cfg.validate_against_rollout_n(4)
    cfg.validate_against_rollout_n(16)
    MixedCutsConfig(enabled=False, n_std=8, n_cuts=8).validate_against_rollout_n(4)  # disabled: no constraint


def test_config_from_omegaconf_block():
    block = OmegaConf.create(
        {
            "enabled": True,
            "n_std": 2,
            "n_cuts": 2,
            "cuts": {
                "k": 7,
                "delta": 0.02,
                "t_warm": 3,
                "empty_set_fallback": "argmax",
                "stats_dir": "/run/cuts_stats",
            },
            "dump_samples_per_step": 1,
        }
    )
    cfg = MixedCutsConfig.from_mapping(block)
    assert cfg.n == 4 and cfg.active and cfg.dump_samples_per_step == 1
    assert cfg.cuts == CutsParams(
        k=7, delta=0.02, t_warm=3, empty_set_fallback="argmax", stats_dir="/run/cuts_stats"
    )
    assert MixedCutsConfig.from_mapping(None).active is False
    with pytest.raises(ValueError):
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 0, "n_cuts": 0})
    with pytest.raises(TypeError):
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 1, "n_cuts": 1, "cuts": {"kk": 5}})
