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


@pytest.fixture(autouse=True)
def _neutral_env(monkeypatch):
    """The env files export MC_HW_PROFILE / MC_SLURM_GPUS (sm75 on Ada, h100 on Jarvislabs); each test sets its own."""
    monkeypatch.delenv("MC_HW_PROFILE", raising=False)
    monkeypatch.delenv("MC_SLURM_GPUS", raising=False)


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
    assert cfg.actor_rollout_ref.actor.checkpoint.save_lora_only is True
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


H100 = ("layout=h100_1gpu", "memory=plan_b_lora_gpu", "hardware=h100")


@pytest.mark.parametrize("name", ["smoke", "math_grpo", "math_mixed_cuts"])
def test_h100_variants_pass_the_checks(compose, name, monkeypatch):
    monkeypatch.setenv("MC_HW_PROFILE", "h100")
    monkeypatch.setenv("MC_SLURM_GPUS", "1")
    mod, cfg = compose(name, *H100)
    a = cfg.actor_rollout_ref
    assert cfg.hw_profile == "h100" and a.rollout.dtype == "bfloat16"
    assert a.actor.fsdp_config.mixed_precision.param_dtype == "bf16"
    assert a.rollout.engine_kwargs.vllm.attention_backend == "FLASH_ATTN"
    assert a.actor.fsdp_config.offload_policy is False and a.model.lora_rank == 64
    assert abs(a.actor.optim.lr - 1e-5) < 1e-12
    assert mod.check(cfg) == []


def test_the_arms_differ_only_in_the_group_split_on_the_h100(compose):
    _mod, grpo = compose("math_grpo", *H100)
    _mod, mixed = compose("math_mixed_cuts", *H100)
    assert (grpo.mixed_cuts.n_std, grpo.mixed_cuts.n_cuts) == (16, 0)
    assert (mixed.mixed_cuts.n_std, mixed.mixed_cuts.n_cuts) == (8, 8)
    grpo.mixed_cuts.n_std, grpo.mixed_cuts.n_cuts = 8, 8
    assert grpo == mixed


def test_learning_rate_follows_the_memory_plan(compose):
    """Until 2026-09-29 base_grpo.yaml's body overrode plan_b_lora's 1e-5 with 1e-6 (decision 013)."""
    _mod, lora = compose("math_grpo")
    _mod, full = compose("math_mixed_cuts", "layout=nlp_4gpu", "memory=plan_a_fullft_offload")
    assert abs(lora.actor_rollout_ref.actor.optim.lr - 1e-5) < 1e-12
    assert abs(full.actor_rollout_ref.actor.optim.lr - 1e-6) < 1e-12


def test_a_learning_rate_that_disagrees_with_the_plan_is_refused(compose):
    mod, cfg = compose("math_grpo", "actor_rollout_ref.actor.optim.lr=1e-6")
    assert any("later config layer" in p for p in mod.check(cfg))


def test_profile_must_match_the_machine(compose, monkeypatch):
    monkeypatch.setenv("MC_HW_PROFILE", "h100")
    mod, cfg = compose("math_grpo")  # the sm75 defaults on an H100 machine
    assert any("MC_HW_PROFILE" in p for p in mod.check(cfg))


def _leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else k)
    else:
        yield prefix, d


@pytest.mark.parametrize(
    ("overrides", "groups"),
    [
        ((), {"layout": "research_1gpu", "memory": "plan_b_lora", "hardware": "sm75"}),
        (
            ("layout=nlp_4gpu", "memory=plan_a_fullft_offload"),
            {"layout": "nlp_4gpu", "memory": "plan_a_fullft_offload"},
        ),
        (("layout=nlp_4gpu", "memory=plan_a_manual_offload"), {"memory": "plan_a_manual_offload"}),
        (H100, {"layout": "h100_1gpu", "memory": "plan_b_lora_gpu", "hardware": "h100"}),
    ],
)
def test_no_group_value_is_silently_overridden(compose, overrides, groups):
    """Every key a layout/memory/hardware file sets must survive composition (the lr bug, decision 013)."""
    import yaml
    from omegaconf import OmegaConf

    _mod, cfg = compose("math_mixed_cuts", *overrides)
    composed = OmegaConf.to_container(cfg, resolve=False)
    shadowed = []
    for group, option in groups.items():
        for key, want in _leaves(yaml.safe_load((REPO / f"configs/train/{group}/{option}.yaml").read_text())):
            got = composed
            for part in key.split("."):
                got = got.get(part) if isinstance(got, dict) else None
            if got != want:
                shadowed.append(f"{group}/{option}: {key} = {want!r}, composed {got!r}")
    assert shadowed == []


def test_select_prints_a_leaf_value(compose, monkeypatch, capsys):
    """jarvis/train.sh reads the step count with --select trainer.total_training_steps."""
    mod, _cfg = compose("math_grpo")
    argv = ["compose_config.py", "math_grpo", *H100, "--select", "trainer.total_training_steps"]
    if _verl_config_dir():
        argv += ["--verl-config-dir", _verl_config_dir()]
    monkeypatch.setattr("sys.argv", argv)
    assert mod.main() == 0
    assert capsys.readouterr().out.strip() == "100"
