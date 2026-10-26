"""scripts/check_recipe.py: a run dir is never resumed under another recipe (decision 018).

Composes real training configs, so it needs verl's config tree like tests/test_layout_guards.py.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

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


pytestmark = pytest.mark.skipif(
    _verl_config_dir() is None and importlib.util.find_spec("verl") is None,
    reason="verl config tree not available (set VERL_CONFIG_DIR to <checkout>/verl/trainer/config)",
)


@pytest.fixture(scope="module")
def tools():
    return _load("check_recipe", "scripts/check_recipe.py"), _load(
        "compose_config", "scripts/compose_config.py"
    )


def _earlier_run(tmp_path: Path, compose_mod, *overrides: str, checkpoint: bool = True) -> Path:
    """A run dir as an earlier submission leaves it: its job's composed config and, optionally, a checkpoint."""
    run = tmp_path / "run"
    cfg = compose_mod.compose_train_config("math_grpo", _verl_config_dir(), list(overrides))
    saved = run / "jobs" / "1001" / "hydra" / ".hydra" / "config.yaml"
    saved.parent.mkdir(parents=True)
    OmegaConf.save(cfg, saved)
    if checkpoint:
        (run / "checkpoints" / "global_step_3").mkdir(parents=True)
        (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("3")
    return run


def _check(check_mod, run: Path, *overrides: str) -> int:
    argv = [str(run), "math_grpo", *overrides]
    if _verl_config_dir():
        argv += ["--verl-config-dir", _verl_config_dir()]
    return check_mod.main(argv)


def test_same_recipe_resumes(tools, tmp_path, capsys):
    check, compose = tools
    run = _earlier_run(tmp_path, compose, "memory=plan_b_lora_calibrated")
    assert _check(check, run, "memory=plan_b_lora_calibrated") == 0
    assert "same recipe" in capsys.readouterr().out


def test_another_recipe_is_refused(tools, tmp_path, capsys, monkeypatch):
    """The September Ada attempt: plan_b_lora (lr 1e-5 constant) under the run name the new recipe would use."""
    monkeypatch.delenv("MC_ALLOW_RECIPE_CHANGE", raising=False)
    check, compose = tools
    run = _earlier_run(tmp_path, compose, "memory=plan_b_lora")
    assert _check(check, run, "memory=plan_b_lora_calibrated") == 5
    out = capsys.readouterr().out
    assert "actor_rollout_ref.actor.optim.lr:" in out and "lr_scheduler_type" in out


def test_mc_allow_recipe_change_resumes_anyway(tools, tmp_path, monkeypatch):
    check, compose = tools
    run = _earlier_run(tmp_path, compose, "memory=plan_b_lora")
    monkeypatch.setenv("MC_ALLOW_RECIPE_CHANGE", "1")
    assert _check(check, run, "memory=plan_b_lora_calibrated") == 0


def test_nothing_to_resume_is_fine(tools, tmp_path):
    check, compose = tools
    run = _earlier_run(tmp_path, compose, "memory=plan_b_lora", checkpoint=False)
    assert _check(check, run, "memory=plan_b_lora_calibrated") == 0
    assert _check(check, tmp_path / "fresh", "memory=plan_b_lora_calibrated") == 0


def test_a_new_step_cap_under_a_cosine_schedule_is_refused(tools, tmp_path, monkeypatch):
    """Decision 017: changing the cap on a resume makes the cosine learning rate jump."""
    monkeypatch.delenv("MC_ALLOW_RECIPE_CHANGE", raising=False)
    check, compose = tools
    run = _earlier_run(tmp_path, compose, "memory=plan_b_lora_calibrated", "trainer.total_training_steps=60")
    assert _check(check, run, "memory=plan_b_lora_calibrated", "trainer.total_training_steps=60") == 0
    assert _check(check, run, "memory=plan_b_lora_calibrated", "trainer.total_training_steps=100") == 5


def test_a_new_step_cap_under_a_constant_schedule_is_fine(tools, tmp_path):
    check, compose = tools
    run = _earlier_run(tmp_path, compose, "memory=plan_b_lora", "trainer.total_training_steps=60")
    assert _check(check, run, "memory=plan_b_lora", "trainer.total_training_steps=100") == 0
