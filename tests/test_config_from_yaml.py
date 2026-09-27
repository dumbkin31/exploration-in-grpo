"""The dataclasses must accept exactly what the YAML configs carry.

The first smoke job on Ada died inside a Ray actor with ``MixedCutsConfig.__init__() got an unexpected
keyword argument 'checks'``: the ``mixed_cuts.checks`` block existed in base_grpo.yaml but not in the
dataclass, and nothing built the dataclass from the composed config before a GPU did. These tests do,
from the literal YAML blocks (always) and from the fully composed configs (when verl's config tree is
available: installed verl, or VERL_CONFIG_DIR/MC_VERL_CONFIG_DIR = <checkout>/verl/trainer/config).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

from mixed_cuts.scheduler import MixedCutsConfig
from mixed_cuts.stability import StabilityConfig

REPO = Path(__file__).resolve().parent.parent


def _yaml_block(name: str, key: str) -> dict:
    raw = yaml.safe_load((REPO / "configs" / "train" / f"{name}.yaml").read_text())
    return raw.get(key) or {}


def test_base_yaml_blocks_build_the_dataclasses():
    mc = _yaml_block("base_grpo", "mixed_cuts")
    mc["cuts"]["stats_dir"] = "/tmp/x"  # the YAML value is a Hydra interpolation
    cfg = MixedCutsConfig.from_mapping(mc)
    assert cfg.group_size == 16 and cfg.n_std == 8 and cfg.n_cuts == 8
    assert cfg.checks.assert_non_thinking is True and cfg.checks.assert_recomputed_logprobs is True
    assert cfg.cuts.k == 5 and abs(cfg.cuts.delta - 0.03) < 1e-12 and cfg.cuts.t_warm == 5
    st = StabilityConfig.from_mapping(_yaml_block("base_grpo", "stability"))
    assert st.grad_norm_max == 50.0 and st.abort_on_nan is False


def test_unknown_mixed_cuts_key_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown keys \\['typo'\\]"):
        MixedCutsConfig.from_mapping({"enabled": True, "n_std": 8, "n_cuts": 8, "typo": 1})


def _verl_config_dir() -> str | None:
    for var in ("VERL_CONFIG_DIR", "MC_VERL_CONFIG_DIR"):
        d = os.environ.get(var)
        if d and Path(d, "ppo_trainer.yaml").exists():
            return d
    return None


@pytest.mark.parametrize("name", ["base_grpo", "smoke", "math_grpo", "math_mixed_cuts"])
def test_composed_configs_build_the_dataclasses(name, monkeypatch):
    if _verl_config_dir() is None and importlib.util.find_spec("verl") is None:
        pytest.skip("verl config tree not available")
    monkeypatch.delenv("MC_SLURM_GPUS", raising=False)
    spec = importlib.util.spec_from_file_location("compose_config", REPO / "scripts" / "compose_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    cfg = mod.compose_train_config(name, _verl_config_dir(), [])
    mc = MixedCutsConfig.from_mapping(cfg.mixed_cuts)
    mc.validate_against_rollout_n(int(cfg.actor_rollout_ref.rollout.n))
    StabilityConfig.from_mapping(cfg.stability)
    assert mc.cuts.stats_dir and mc.cuts.stats_dir.endswith("cuts_stats")
