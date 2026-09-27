"""check_resume must find the W&B run id for online (run-*) and offline (offline-run-*) folders alike."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _check_resume():
    spec = importlib.util.spec_from_file_location("check_resume", REPO / "scripts" / "check_resume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_online_and_offline_folders_share_one_id(tmp_path: Path):
    (tmp_path / "wandb" / "run-20260927_165202-math_grpo-s1").mkdir(parents=True)
    (tmp_path / "wandb" / "offline-run-20260928_010101-math_grpo-s1").mkdir(parents=True)
    assert _check_resume().wandb_run_ids(tmp_path) == {"math_grpo-s1"}


def test_legacy_nested_wandb_dir_is_still_found(tmp_path: Path):
    (tmp_path / "wandb" / "wandb" / "offline-run-20260927_165202-abc123").mkdir(parents=True)
    assert _check_resume().wandb_run_ids(tmp_path) == {"abc123"}


def test_two_ids_are_two_ids(tmp_path: Path):
    (tmp_path / "wandb" / "run-20260927_1-one").mkdir(parents=True)
    (tmp_path / "wandb" / "run-20260927_2-two").mkdir(parents=True)
    assert _check_resume().wandb_run_ids(tmp_path) == {"one", "two"}
