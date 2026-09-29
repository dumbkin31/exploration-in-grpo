"""configs/ada.env.sh hands over to configs/jarvis.env.sh when MC_SITE=jarvis (decision 013)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

KEYS = (
    "MC_SITE",
    "MC_HW_PROFILE",
    "MC_LAYOUT",
    "MC_TRAIN_OVERRIDES",
    "MC_CHECKPOINT_HOME",
    "MC_RUN_TAG",
    "WANDB_PROJECT",
)


def _source(tmp_path: Path, **env: str) -> dict[str, str]:
    script = f'source "{REPO}/configs/ada.env.sh" >/dev/null 2>&1; ' + "; ".join(
        f'printf "%s=%s\\n" {k} "${{{k}:-}}"' for k in KEYS
    )
    base = {"HOME": str(tmp_path), "PATH": os.environ["PATH"], "USER": "u"}
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env={**base, **env}, timeout=60
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def test_ada_is_the_default_and_unchanged(tmp_path: Path):
    v = _source(tmp_path)
    assert v["MC_SITE"] == "ada" and v["MC_HW_PROFILE"] == "sm75"
    assert v["MC_LAYOUT"] == "research_1gpu" and v["MC_TRAIN_OVERRIDES"] == ""
    assert v["MC_RUN_TAG"] == "", "Ada run names stay <config>-s<seed>"


def test_jarvis_selects_the_h100_setup(tmp_path: Path):
    v = _source(
        tmp_path,
        MC_SITE="jarvis",
        MC_STAGE_ROOT=str(tmp_path / "stage"),
        MC_SCRATCH_ROOT=str(tmp_path / "scratch"),
        UV_PYTHON_INSTALL_DIR=str(tmp_path / "uvpy"),
    )
    assert v["MC_SITE"] == "jarvis" and v["MC_HW_PROFILE"] == "h100" and v["MC_LAYOUT"] == "h100_1gpu"
    assert v["MC_TRAIN_OVERRIDES"] == "layout=h100_1gpu memory=plan_b_lora_gpu hardware=h100"
    assert v["MC_CHECKPOINT_HOME"] == "durable"
    assert v["MC_RUN_TAG"] == "h100", "run names math_grpo-h100-s1 / math_mixed_cuts-h100-s1"
    assert v["WANDB_PROJECT"] == "mixed-cuts"


def test_kaggle_selects_the_t4_setup(tmp_path: Path):
    v = _source(
        tmp_path,
        MC_SITE="kaggle",
        MC_STAGE_ROOT=str(tmp_path / "stage"),
        MC_SCRATCH_ROOT=str(tmp_path / "scratch"),
        UV_PYTHON_INSTALL_DIR=str(tmp_path / "uvpy"),
    )
    assert v["MC_SITE"] == "kaggle" and v["MC_HW_PROFILE"] == "sm75" and v["MC_LAYOUT"] == "kaggle_t4"
    assert v["MC_TRAIN_OVERRIDES"] == "layout=kaggle_t4", (
        "the Ada LoRA plan and fp16 profile are the defaults"
    )
    assert v["MC_CHECKPOINT_HOME"] == "durable" and v["MC_RUN_TAG"] == "t4"


def test_preflight_profiles():
    sys.path.insert(0, str(REPO / "scripts"))
    import check_env

    assert check_env.PROFILES["h100"]["cc"] == (9, 0) and check_env.PROFILES["h100"]["bf16"] is True
    assert check_env.PROFILES["sm75"]["backend"] == "TRITON_ATTN"
