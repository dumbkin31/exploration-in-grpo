"""Layout guards (decision 012): the 1-GPU research layout must refuse plan A, accept plan B and the
4-GPU layout with plan A. Needs verl's config tree: installed verl, or VERL_CONFIG_DIR/MC_VERL_CONFIG_DIR
pointing at <checkout>/verl/trainer/config; skipped otherwise."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _verl_config_dir() -> str | None:
    for var in ("VERL_CONFIG_DIR", "MC_VERL_CONFIG_DIR"):
        d = os.environ.get(var)
        if d and Path(d, "ppo_trainer.yaml").exists():
            return d
    return None


def _can_compose() -> bool:
    return _verl_config_dir() is not None or importlib.util.find_spec("verl") is not None


@pytest.fixture(scope="module")
def compose():
    if not _can_compose():
        pytest.skip("verl config tree not available (set VERL_CONFIG_DIR to <checkout>/verl/trainer/config)")
    mod = _load("compose_config", "scripts/compose_config.py")
    d = _verl_config_dir()

    def _compose(name: str, *overrides: str):
        return mod, mod.compose_train_config(name, d, list(overrides))

    return _compose


@pytest.mark.parametrize("name", ["base_grpo", "smoke", "math_grpo", "math_mixed_cuts"])
def test_defaults_are_the_1gpu_lora_layout(compose, name, monkeypatch):
    monkeypatch.delenv("MC_SLURM_GPUS", raising=False)
    mod, cfg = compose(name)
    assert cfg.trainer.n_gpus_per_node == 1
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert cfg.actor_rollout_ref.model.lora_rank == 64
    assert cfg.actor_rollout_ref.actor.fsdp_config.offload_policy is True
    assert mod.check(cfg) == []


def test_1gpu_plus_plan_a_is_refused(compose, monkeypatch):
    monkeypatch.delenv("MC_SLURM_GPUS", raising=False)
    mod, cfg = compose("math_mixed_cuts", "memory=plan_a_fullft_offload")
    problems = mod.check(cfg)
    assert any("LoRA" in p for p in problems), problems


def test_4gpu_layout_with_plan_a_still_composes(compose, monkeypatch):
    monkeypatch.delenv("MC_SLURM_GPUS", raising=False)
    mod, cfg = compose("math_mixed_cuts", "layout=nlp_4gpu", "memory=plan_a_fullft_offload")
    assert cfg.trainer.n_gpus_per_node == 4
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 4
    assert mod.check(cfg) == []


def test_sbatch_gpu_count_mismatch_is_caught(compose, monkeypatch):
    monkeypatch.setenv("MC_SLURM_GPUS", "4")
    mod, cfg = compose("math_mixed_cuts")
    assert any("MC_SLURM_GPUS" in p for p in mod.check(cfg))


def test_smoke_save_freq_follows_the_top_level_knob(compose, monkeypatch):
    monkeypatch.delenv("MC_SLURM_GPUS", raising=False)
    _mod, cfg = compose("smoke", "save_freq=5")
    assert cfg.trainer.save_freq == 5
