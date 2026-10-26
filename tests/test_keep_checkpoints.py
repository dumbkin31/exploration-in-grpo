"""slurm/common.sh::mc_keep_checkpoints keeps every 10th checkpoint that verl would prune (decision 018)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _checkpoint(run: Path, step: int) -> Path:
    """The layout verl v0.9.0 writes for a LoRA actor (as in the H200 runs)."""
    d = run / "checkpoints" / f"global_step_{step}"
    (d / "actor" / "huggingface").mkdir(parents=True)
    (d / "actor" / "model_world_size_1_rank_0.pt").write_bytes(b"weights %d" % step)
    (d / "actor" / "optim_world_size_1_rank_0.pt").write_bytes(b"adam state")
    (d / "actor" / "extra_state_world_size_1_rank_0.pt").write_bytes(b"rng + scheduler")
    (d / "actor" / "huggingface" / "config.json").write_text("{}")
    (d / "data.pt").write_bytes(b"dataloader")
    return d


def _keep(run: Path, **env: str) -> subprocess.CompletedProcess:
    base = {"PATH": os.environ["PATH"], "HOME": str(run.parent), "MC_RUN_DIR": str(run)}
    script = f'set -euo pipefail; source "{REPO}/slurm/common.sh"; mc_keep_checkpoints; echo KEEP_OK'
    p = subprocess.run(
        ["bash", "-c", script], env={**base, **env}, capture_output=True, text=True, timeout=60
    )
    assert p.returncode == 0 and "KEEP_OK" in p.stdout, p.stdout + p.stderr
    return p


def _latest(run: Path, step: int) -> None:
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(str(step))


def test_every_tenth_completed_checkpoint_is_kept_without_the_optimizer(tmp_path: Path):
    run = tmp_path / "run"
    for step in (9, 10, 11, 20):
        _checkpoint(run, step)
    _latest(run, 11)  # step 20 is still being written
    _keep(run)
    kept = run / "checkpoints_kept"
    assert sorted(p.name for p in kept.iterdir()) == ["global_step_10"]
    k = kept / "global_step_10"
    assert (k / "actor" / "model_world_size_1_rank_0.pt").read_bytes() == b"weights 10"
    assert (k / "actor" / "huggingface" / "config.json").exists() and (k / "data.pt").exists()
    assert not (k / "actor" / "optim_world_size_1_rank_0.pt").exists()
    # hard links: no extra space while verl still holds the original
    orig = run / "checkpoints" / "global_step_10" / "actor" / "model_world_size_1_rank_0.pt"
    assert (k / "actor" / "model_world_size_1_rank_0.pt").stat().st_ino == orig.stat().st_ino


def test_kept_weights_survive_verl_pruning_and_later_steps_are_added(tmp_path: Path):
    run = tmp_path / "run"
    for step in (10, 20):
        _checkpoint(run, step)
    _latest(run, 10)
    _keep(run)
    shutil.rmtree(run / "checkpoints" / "global_step_10")  # verl's max_actor_ckpt_to_keep
    _latest(run, 20)
    p = _keep(run)
    assert "kept checkpoint global_step_20" in p.stdout and "global_step_10" not in p.stdout  # idempotent
    kept = run / "checkpoints_kept"
    assert (kept / "global_step_10" / "actor" / "model_world_size_1_rank_0.pt").read_bytes() == b"weights 10"
    assert (kept / "global_step_20" / "actor" / "model_world_size_1_rank_0.pt").read_bytes() == b"weights 20"
    assert not any(p.name.endswith(".partial") for p in kept.iterdir())


def test_disabled_or_nothing_to_keep(tmp_path: Path):
    run = tmp_path / "run"
    _keep(run)  # no checkpoints dir yet
    _checkpoint(run, 10)
    _latest(run, 10)
    _keep(run, MC_KEEP_EVERY="0")
    assert not (run / "checkpoints_kept").exists()
    _keep(run, MC_KEEP_EVERY="5")
    assert (run / "checkpoints_kept" / "global_step_10").is_dir()
