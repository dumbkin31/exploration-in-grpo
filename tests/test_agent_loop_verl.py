"""verl-dependent tests of the rollout-worker override (skipped in the CPU dev env).

They build *bare* worker instances (no Ray, no TransferQueue, no LLM server) by bypassing
``__init__`` and stubbing the two collaborators ``_run_prompt`` touches:
``transfer_queue.async_kv_put`` and ``_run_agent_loop``. The key assertion: with ``n_cuts = 0``
our ``_run_prompt`` issues *exactly* the calls verl's own ``AgentLoopWorkerTQ._run_prompt`` issues.
"""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import OmegaConf

from tests.conftest import HAS_VERL, needs_verl

pytestmark = [needs_verl, pytest.mark.needs_verl]

if HAS_VERL:
    from verl.trainer.ppo.v1 import agent_loop_tq as verl_tq

    from mixed_cuts import agent_loop as mc_agent_loop
    from mixed_cuts.agent_loop import MixedCutsAgentLoopWorkerTQBase, _unwrap_ray_actor_class

BASE_SP = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}


def _config(n: int, n_std: int, n_cuts: int, enabled: bool = True):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": n, "val_kwargs": {"n": 2}}},
            "mixed_cuts": {
                "enabled": enabled,
                "n_std": n_std,
                "n_cuts": n_cuts,
                "group_size": n if enabled else None,
                "cuts": {"k": 5, "delta": 0.03, "t_warm": 5, "stats_dir": "/tmp/cuts_stats"},
            },
        }
    )


class Recorder:
    def __init__(self):
        self.calls: list[tuple[dict, dict]] = []

    async def __call__(self, sampling_params, **kwargs):
        self.calls.append((dict(sampling_params), dict(kwargs)))


def _bare(cls, config):
    obj = object.__new__(cls)
    obj.config = config
    if hasattr(obj, "_init_mixed_cuts"):
        obj._init_mixed_cuts()
    return obj


async def _noop_put(**kwargs):
    return None


def _run(worker, prompt, validate=False):
    rec = Recorder()
    worker._run_agent_loop = rec
    traj = {"step": 3, "sample_index": 0, "rollout_n": 0, "validate": validate}
    asyncio.run(worker._run_prompt(dict(prompt), dict(BASE_SP), traj, trace=False))
    return rec.calls


def test_n_cuts_zero_is_call_for_call_identical_to_verl(monkeypatch):
    monkeypatch.setattr(verl_tq.tq, "async_kv_put", _noop_put)
    monkeypatch.setattr(mc_agent_loop.tq, "async_kv_put", _noop_put)
    base_cls = _unwrap_ray_actor_class(verl_tq.AgentLoopWorkerTQ)
    cfg = _config(n=4, n_std=4, n_cuts=0)
    prompt = {
        "uid": "u1",
        "raw_prompt": [{"role": "user", "content": "1+1?"}],
        "agent_name": "single_turn_agent",
    }

    theirs = _run(_bare(base_cls, cfg), prompt)
    ours = _run(_bare(MixedCutsAgentLoopWorkerTQBase, cfg), prompt)
    assert len(theirs) == len(ours) == 4
    for (sp_t, kw_t), (sp_o, kw_o) in zip(theirs, ours, strict=True):
        assert sp_t == sp_o, "sampling params must be byte-identical with n_cuts=0"
        assert kw_o.pop("rollout_kind") == "std"
        assert kw_t == kw_o, "all other kwargs (session_id, prompt fields) must match verl"

    # disabled block: same story even when n_std/n_cuts look mixed
    ours_disabled = _run(_bare(MixedCutsAgentLoopWorkerTQBase, _config(4, 2, 2, enabled=False)), prompt)
    assert [sp for sp, _ in ours_disabled] == [sp for sp, _ in theirs]


def test_mixed_group_tags_and_extra_args(monkeypatch):
    monkeypatch.setattr(mc_agent_loop.tq, "async_kv_put", _noop_put)
    worker = _bare(MixedCutsAgentLoopWorkerTQBase, _config(n=4, n_std=1, n_cuts=3))
    calls = _run(worker, {"uid": "u9", "raw_prompt": [], "agent_name": "single_turn_agent"})
    kinds = [kw["rollout_kind"] for _, kw in calls]
    assert kinds == ["std", "cuts", "cuts", "cuts"]
    assert [kw["session_id"] for _, kw in calls] == [0, 1, 2, 3]
    assert "extra_args" not in calls[0][0]
    for sp, kw in calls[1:]:
        c = sp["extra_args"]["cuts"]
        assert c["uid"] == "u9" and c["session_id"] == kw["session_id"] and c["step"] == 3
    # validation prompts are always standard
    val_calls = _run(worker, {"uid": "v", "raw_prompt": [], "agent_name": "single_turn_agent"}, validate=True)
    assert [kw["rollout_kind"] for _, kw in val_calls] == ["std", "std"]  # val_kwargs.n == 2


def test_group_size_mismatch_fails_at_init():
    with pytest.raises(ValueError, match="budget mismatch"):
        _bare(MixedCutsAgentLoopWorkerTQBase, _config(n=16, n_std=4, n_cuts=4))


def test_full_group_8_8_through_the_worker(monkeypatch):
    """Task B (mocked engine): the worker issues exactly 8 CUTS + 8 standard requests sharing the uid."""
    monkeypatch.setattr(mc_agent_loop.tq, "async_kv_put", _noop_put)
    worker = _bare(MixedCutsAgentLoopWorkerTQBase, _config(n=16, n_std=8, n_cuts=8))
    calls = _run(worker, {"uid": "p1", "raw_prompt": [], "agent_name": "single_turn_agent"})
    assert len(calls) == 16
    cuts_calls = [(sp, kw) for sp, kw in calls if "cuts" in (sp.get("extra_args") or {})]
    std_calls = [(sp, kw) for sp, kw in calls if "cuts" not in (sp.get("extra_args") or {})]
    assert len(cuts_calls) == 8 and len(std_calls) == 8
    assert all(kw["uid"] == "p1" for _, kw in calls), "all 16 sessions share the prompt uid (one GRPO group)"
    assert all(kw["rollout_kind"] == "cuts" for _, kw in cuts_calls)
    assert all(kw["rollout_kind"] == "std" for _, kw in std_calls)
    assert all(sp["top_p"] == 1.0 and sp["top_k"] == -1 for sp, _ in cuts_calls)
    assert sorted(kw["session_id"] for _, kw in calls) == list(range(16))
