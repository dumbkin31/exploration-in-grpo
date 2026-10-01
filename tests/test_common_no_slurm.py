"""slurm/common.sh's stage-out must work on machines without SLURM (Jarvislabs, Kaggle).

The first H200 smoke run finished training with rc=0, then the EXIT-trap stage-out died on the missing
`sstat` under `set -euo pipefail`, so the job looked failed and the GPU tests never ran.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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
