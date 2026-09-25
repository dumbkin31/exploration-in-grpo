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
    assert cfg.group_size == 16  # defaults to n_std + n_cuts
    with pytest.raises(ValueError, match="G = n_std \\+ n_cuts"):
        plan_group(BASE, 4, cfg, uid="u", step=0)
    with pytest.raises(ValueError, match="budget mismatch"):
        cfg.validate_against_rollout_n(4)
    cfg.validate_against_rollout_n(16)
    MixedCutsConfig(enabled=False, n_std=8, n_cuts=8).validate_against_rollout_n(4)  # disabled: no constraint


def test_d4_budget_assertion_exact_message():
    """n_std + n_cuts == G always; the vanilla arm (16 + 0) costs the same as the mixed arm (8 + 8)."""
    with pytest.raises(ValueError, match=r"^budget mismatch: 8 \+ 4 != G=16$"):
        MixedCutsConfig(enabled=True, n_std=8, n_cuts=4, group_size=16)
    MixedCutsConfig(enabled=True, n_std=16, n_cuts=0, group_size=16)
    MixedCutsConfig(enabled=True, n_std=8, n_cuts=8, group_size=16)
    MixedCutsConfig(enabled=False, n_std=8, n_cuts=4, group_size=16)  # disabled: no constraint


def test_full_group_8_8_carries_exactly_8_cuts_payloads():
    """Task B (CPU layer): with n_cuts=8, exactly 8 of 16 requests carry the CUTS extra_args."""
    from mixed_cuts.diagnostics import compute_group_diagnostics

    cfg = MixedCutsConfig(enabled=True, n_std=8, n_cuts=8, group_size=16)
    specs = plan_group(BASE, 16, cfg, uid="prompt-42", step=1)
    assert len(specs) == 16
    with_payload = [s for s in specs if EXTRA_ARGS_KEY in (s.sampling_params.get("extra_args") or {})]
    without = [s for s in specs if EXTRA_ARGS_KEY not in (s.sampling_params.get("extra_args") or {})]
    assert len(with_payload) == 8 and len(without) == 8
    assert [s.rollout_kind for s in with_payload] == [CUTS] * 8 and [s.rollout_kind for s in without] == [
        STD
    ] * 8
    assert sorted(s.session_id for s in specs) == list(range(16))
    for s in with_payload:
        assert CutsParams.from_extra_args(s.sampling_params["extra_args"]).uid == "prompt-42"
    # Reassembled into ONE group before advantage normalisation: all 16 rollouts share the prompt uid.
    metrics = compute_group_diagnostics(
        ["prompt-42"] * 16, [1.0] * 8 + [0.0] * 8, [s.rollout_kind for s in specs]
    )
    assert metrics["mixed_cuts/n_groups"] == 1 and metrics["mixed_cuts/group_size_mean"] == 16


def test_cuts_sessions_force_top_p_one_and_top_k_off():
    """D2: vLLM's top-k/top-p run after the processor and would re-narrow the uniform set."""
    base = {**BASE, "top_p": 0.8, "top_k": 20}
    cfg = MixedCutsConfig(enabled=True, n_std=1, n_cuts=1)
    std, cuts = plan_group(base, 2, cfg, uid="u", step=0)
    assert (
        std.sampling_params["top_p"] == 0.8 and std.sampling_params["top_k"] == 20
    )  # standard arm untouched
    assert cuts.sampling_params["top_p"] == 1.0 and cuts.sampling_params["top_k"] == -1


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
    assert cfg.n == 4 and cfg.group_size == 4 and cfg.active and cfg.dump_samples_per_step == 1
    assert (
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 2, "n_cuts": 2, "group_size": 4}).group_size
        == 4
    )
    with pytest.raises(ValueError, match="budget mismatch: 2 \\+ 2 != G=8"):
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 2, "n_cuts": 2, "group_size": 8})
    assert cfg.cuts == CutsParams(
        k=7, delta=0.02, t_warm=3, empty_set_fallback="argmax", stats_dir="/run/cuts_stats"
    )
    assert MixedCutsConfig.from_mapping(None).active is False
    with pytest.raises(ValueError):
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 0, "n_cuts": 0})
    with pytest.raises(TypeError):
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 1, "n_cuts": 1, "cuts": {"kk": 5}})
