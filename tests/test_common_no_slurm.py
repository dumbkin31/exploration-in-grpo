"""slurm/common.sh's stage-out must work on machines without SLURM (Jarvislabs, Kaggle).

The first H200 smoke run finished training with rc=0, then the EXIT-trap stage-out died on the missing
`sstat` under `set -euo pipefail`, so the job looked failed and the GPU tests never ran.
A job that stops before its stage-out trap is installed (e.g. a failed config preflight) must not leave
its GPU memory sampler running.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_stage_out_without_slurm_tools(tmp_path: Path):
    run = tmp_path / "run"
    job = run / "jobs" / "j1"
    job.mkdir(parents=True)
    path = os.pathsep.join(
        p for p in os.environ["PATH"].split(os.pathsep) if not (Path(p) / "sstat").exists()
    )
    env = {
        "PATH": path,
        "HOME": str(tmp_path),
        "MC_RUN_DIR": str(run),
        "MC_DURABLE_DIR": str(run),  # durable mode: the mirror is a no-op
        "MC_JOB_DIR": str(job),
        "MC_JOB_ID": "j1",
        "MC_JOB_STDOUT": str(tmp_path / "missing.out"),
    }
    script = f'set -euo pipefail; source "{REPO}/slurm/common.sh"; mc_stage_out; echo STAGE_OUT_OK'
    p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "STAGE_OUT_OK" in p.stdout, p.stdout + p.stderr
    assert (job / "host_mem_peak.txt").read_text().count("sstat_maxrss_batch=unknown") == 1


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# Ways mc_train_main stops between mc_job_init (which starts the sampler) and mc_run_with_traps (stage-out):
# a bash stub, the exit code of the config preflight (scripts/check_env.py), the job's exit code and message.
EARLY_EXITS = {
    "config preflight fails": ("mc_check_recipe() { :; }", 1, 4, "config preflight FAILED"),
    # the run dir has checkpoints but no recipe.txt
    "recipe guard refuses": ("mc_recipe_fingerprint() { echo r; }", 0, 7, "recipe of those checkpoints"),
}


@pytest.mark.parametrize("early_exit", list(EARLY_EXITS))
def test_an_early_exit_stops_the_gpu_sampler(tmp_path: Path, early_exit: str):
    stub, preflight_rc, job_rc, message = EARLY_EXITS[early_exit]
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    # stand-ins: a sampler that never exits by itself, and the config preflight
    (repo / "scripts" / "profile_memory.py").write_text("import time\ntime.sleep(300)\n")
    (repo / "scripts" / "check_env.py").write_text(f"import sys\nsys.exit({preflight_rc})\n")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").symlink_to(sys.executable)
    job = tmp_path / "run" / "jobs" / "j1"
    job.mkdir(parents=True)
    (tmp_path / "run" / "checkpoints").mkdir()
    (tmp_path / "run" / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("72\n")
    env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "MC_REPO_ROOT": str(repo),
        "MC_RUN_DIR": str(tmp_path / "run"),
        "MC_RUN_NAME": "math_grpo-s1",
        "MC_JOB_DIR": str(job),
        "MC_JOB_ID": "j1",
        "MC_STAGED_MODEL_DIR": "model",
    }
    script = f"""set -euo pipefail
source "{REPO}/slurm/common.sh"
mc_job_init() {{ mc_start_gpu_sampler; echo "SAMPLER ${{_MC_SAMPLER_PID}}"; }}   # its last step, for real
mc_stage_in() {{ :; }}
{stub}
mc_train_main math_grpo"""
    p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    pid = int(re.search(r"SAMPLER (\d+)", p.stdout).group(1))
    alive = _alive(pid)
    if alive:
        os.kill(pid, 9)
    assert p.returncode == job_rc and message in p.stdout, p.stdout + p.stderr
    assert not alive, "the GPU sampler outlived the job"


def test_a_failed_job_preflight_exits_4_with_its_message(tmp_path: Path):
    # mc_job_init's preflight is a `python ... | tee` pipeline: under set -e + pipefail a failure used to end
    # the job with rc 1 before its `test "${PIPESTATUS[0]}"` check could log and exit 4
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "scripts" / "check_env.py").write_text("import sys\nsys.exit(1)\n")
    (repo / "configs" / "ada.env.sh").write_text(  # a stand-in site file: every path under tmp_path
        f'export MC_STAGE_ROOT="{tmp_path}" MC_RUNS_DIR="{tmp_path}/runs" MC_CHECKPOINT_HOME=durable\n'
        f'export MC_SCRATCH_ROOT="{tmp_path}/scratch" MC_SCRATCH_RUNS_DIR="{tmp_path}/scratch/runs"\n'
        f'export MC_MODEL_DIR="{tmp_path}/model" MC_VENV_DIR="{tmp_path}/venv"\n'
    )
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").symlink_to(sys.executable)
    env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "MC_REPO_ROOT": str(repo),
    }
    script = f"""set -euo pipefail
source "{REPO}/slurm/common.sh"
mc_check_driver() {{ :; }}   # no NVIDIA driver or GPU needed here
mc_check_cuda() {{ :; }}
mc_job_init math_grpo-s1"""
    p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 4 and "preflight FAILED; see" in p.stdout, p.stdout + p.stderr
