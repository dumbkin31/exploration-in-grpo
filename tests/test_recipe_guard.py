"""slurm/common.sh::mc_check_recipe refuses to resume a run dir's checkpoints under another training recipe.

resume_mode=auto continues whatever checkpoints a run dir holds, and the Jarvislabs instance sets
MC_RUN_TAG=h200: relaunching seed 1 under decision 017's recipes would have resumed the diverged LoRA runs
(a full fine-tune cannot load their LoRA-only checkpoints; the calibrated LoRA would continue their weights).
Composing the real recipe needs verl's config tree, so most tests replace mc_recipe_fingerprint with a fake
that prints MC_TRAIN_OVERRIDES and the arguments the guard was called with.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN_A = "layout=h100_1gpu memory=plan_a_fullft_gpu hardware=h100"
H200_LORA = "layout=h100_1gpu memory=plan_b_lora_gpu hardware=h100"  # the diverged H200 runs (016)
FAKE_FINGERPRINT = """mc_recipe_fingerprint() {
  echo "MC_TRAIN_OVERRIDES=${MC_TRAIN_OVERRIDES}"
  echo "args=$*"
}"""


def _guard(tmp_path: Path, run: Path, overrides: str, call: str = "mc_check_recipe math_grpo", **env):
    run.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "MC_RUN_DIR": str(run),
        "MC_RUN_NAME": run.name,
        "MC_RUN_TAG": "h200",
        "MC_JOB_ID": "j1",
        "MC_TRAIN_OVERRIDES": overrides,
        **env,
    }
    script = "\n".join(
        ["set -euo pipefail", f'source "{REPO}/slurm/common.sh"', FAKE_FINGERPRINT, call, "echo GUARD_PASSED"]
    )
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)


def _checkpoint(run: Path, step: int) -> None:
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(f"{step}\n")


def _passed(p) -> bool:
    return p.returncode == 0 and "GUARD_PASSED" in p.stdout


def test_first_start_records_the_recipe(tmp_path: Path):
    run = tmp_path / "math_grpo-h200r2-s1"
    p = _guard(tmp_path, run, PLAN_A)
    assert _passed(p), p.stdout + p.stderr
    lines = (run / "recipe.txt").read_text().splitlines()
    assert lines[0].startswith("#") and lines[1].startswith("#")
    assert lines[2:] == [f"MC_TRAIN_OVERRIDES={PLAN_A}", "args=math_grpo"]


def test_resume_with_the_same_recipe_proceeds(tmp_path: Path):
    run = tmp_path / "math_grpo-h200r2-s1"
    assert _passed(_guard(tmp_path, run, PLAN_A))
    _checkpoint(run, 3)
    first = (run / "recipe.txt").read_text()
    p = _guard(tmp_path, run, PLAN_A, MC_JOB_ID="j2")
    assert _passed(p), p.stdout + p.stderr
    assert (run / "recipe.txt").read_text() == first, "the first start's record (job, date) is kept"


def test_resume_under_another_recipe_is_refused(tmp_path: Path):
    run = tmp_path / "math_grpo-h200-s1"
    assert _passed(_guard(tmp_path, run, H200_LORA))
    _checkpoint(run, 72)
    first = (run / "recipe.txt").read_text()
    p = _guard(tmp_path, run, PLAN_A)
    assert p.returncode == 7 and "GUARD_PASSED" not in p.stdout, p.stdout + p.stderr
    assert "checkpoints up to step 72 from another training recipe" in p.stdout
    assert f"MC_TRAIN_OVERRIDES={H200_LORA}" in p.stdout, "names the recipe of the checkpoints"
    assert f"MC_TRAIN_OVERRIDES={PLAN_A}" in p.stdout, "names the recipe of this launch"
    assert "set a new MC_RUN_TAG (now 'h200'" in p.stdout
    assert (run / "recipe.txt").read_text() == first


def test_checkpoints_without_a_recipe_are_refused(tmp_path: Path):
    run = tmp_path / "math_grpo-h200-s1"  # a run dir from before the guard: checkpoints, no recipe.txt
    _checkpoint(run, 72)
    p = _guard(tmp_path, run, PLAN_A)
    assert p.returncode == 7 and "GUARD_PASSED" not in p.stdout, p.stdout + p.stderr
    assert f"no {run / 'recipe.txt'}" in p.stdout
    assert "recipe of those checkpoints is unknown" in p.stdout
    assert not (run / "recipe.txt").exists()

    p = _guard(tmp_path, run, PLAN_A, MC_ALLOW_RECIPE_CHANGE="1")
    assert _passed(p), p.stdout + p.stderr
    assert f"MC_TRAIN_OVERRIDES={PLAN_A}" in (run / "recipe.txt").read_text()


def test_allowed_recipe_change_keeps_the_old_recipe(tmp_path: Path):
    run = tmp_path / "math_grpo-h200-s1"
    assert _passed(_guard(tmp_path, run, H200_LORA))
    _checkpoint(run, 40)
    p = _guard(tmp_path, run, PLAN_A, MC_ALLOW_RECIPE_CHANGE="1")
    assert _passed(p), p.stdout + p.stderr
    assert f"MC_TRAIN_OVERRIDES={PLAN_A}" in (run / "recipe.txt").read_text()
    assert f"MC_TRAIN_OVERRIDES={H200_LORA}" in (run / "recipe.until-step-40.txt").read_text()


def test_before_the_first_checkpoint_a_relaunch_may_change_the_recipe(tmp_path: Path):
    run = tmp_path / "math_grpo-h200r2-s1"  # e.g. the first attempt crashed before step 1 was saved
    assert _passed(_guard(tmp_path, run, H200_LORA))
    p = _guard(tmp_path, run, PLAN_A)
    assert _passed(p), p.stdout + p.stderr
    assert f"MC_TRAIN_OVERRIDES={PLAN_A}" in (run / "recipe.txt").read_text()


def test_a_recipe_that_does_not_compose_stops_the_job(tmp_path: Path):
    run = tmp_path / "math_grpo-h200r2-s1"
    p = _guard(tmp_path, run, PLAN_A, "mc_recipe_fingerprint() { return 1; }; mc_check_recipe math_grpo")
    assert p.returncode == 4 and "cannot compose the training recipe" in p.stdout, p.stdout + p.stderr
    assert not (run / "recipe.txt").exists()


def test_train_main_checks_the_recipe_before_starting_the_trainer(tmp_path: Path):
    run = tmp_path / "math_grpo-h200-s1"
    _checkpoint(run, 72)
    (tmp_path / "job").mkdir()
    stubs = """
mc_job_init() { :; }        # the paths come from the environment below
mc_stage_in() { :; }
python() { :; }             # the config preflight and the memory report
mc_run_with_traps() { echo "TRAINER STARTED"; }
mc_train_main math_grpo extra.key=1"""
    env = {
        "MC_JOB_DIR": str(tmp_path / "job"),
        "MC_STAGED_MODEL_DIR": "model",
        "MC_STAGED_DATA_DIR": "data",
        "MC_TOTAL_STEPS": "100",
    }
    p = _guard(tmp_path, run, PLAN_A, stubs, **env)
    assert p.returncode == 7 and "TRAINER STARTED" not in p.stdout, p.stdout + p.stderr

    p = _guard(tmp_path, run, PLAN_A, stubs, MC_ALLOW_RECIPE_CHANGE="1", **env)
    assert "TRAINER STARTED" in p.stdout and _passed(p), p.stdout + p.stderr
    # the guard composes with the trainer's overrides: MC_TOTAL_STEPS, MC_TRAIN_OVERRIDES, then the CLI ones
    trainer_overrides = f"trainer.total_training_steps=100 {PLAN_A} extra.key=1"
    assert f"args=math_grpo {trainer_overrides}" in (run / "recipe.txt").read_text()


def _verl_config_dir() -> str | None:  # as in tests/test_layout_guards.py
    for var in ("VERL_CONFIG_DIR", "MC_VERL_CONFIG_DIR"):
        d = os.environ.get(var)
        if d and Path(d, "ppo_trainer.yaml").exists():
            return d
    return None


@pytest.mark.parametrize(
    "plan, lora_rank, lr, schedule",
    [
        ("plan_a_fullft_gpu", "0", "1e-06", "constant"),
        ("plan_b_lora_calibrated_gpu", "64", "2e-06", "cosine"),
        ("plan_b_lora_gpu", "64", "1e-05", "constant"),
    ],
)
def test_real_fingerprint_of_each_jarvis_plan(tmp_path: Path, plan, lora_rank, lr, schedule):
    verl_dir = _verl_config_dir()
    has_verl = verl_dir is not None or importlib.util.find_spec("verl") is not None
    if importlib.util.find_spec("hydra") is None or not has_verl:
        pytest.skip("needs hydra and verl's config tree (installed verl, or VERL_CONFIG_DIR=<checkout>/...)")
    overrides = f"layout=h100_1gpu memory={plan} hardware=h100"
    # without an installed verl, point compose_config.py at the checkout (the trainer always uses installed verl)
    shim = f'python() {{ command python "$@" --verl-config-dir "{verl_dir}"; }}' if verl_dir else ""
    script = f'set -euo pipefail; source "{REPO}/slurm/common.sh"; {shim}\nmc_recipe_fingerprint math_grpo {overrides}'
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",  # `python` = this interpreter
        "MC_REPO_ROOT": str(REPO),
        "MC_TRAIN_OVERRIDES": overrides,
    }
    p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines() == [
        f"MC_TRAIN_OVERRIDES={overrides}",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={schedule}",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl+",
    ]
