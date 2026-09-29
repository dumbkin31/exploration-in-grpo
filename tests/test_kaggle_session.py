"""kaggle/train.sh: resume from the Hub, push each checkpoint, stop before the session limit (decision 014).

Runs the real script in a copy of the repo with a stand-in trainer (one step every second) and a
stand-in hub_sync that records its calls.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

TRAINER = """#!/usr/bin/env bash
d="${MC_RUNS_DIR}/${MC_RUN_NAME}"; mkdir -p "$d/checkpoints"
trap 'exit 143' USR1
s=$(cat "$d/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
while [ "$s" -lt "${MC_TOTAL_STEPS}" ]; do
  sleep 1 & wait $!
  s=$((s+1)); echo "$s" > "$d/checkpoints/latest_checkpointed_iteration.txt"
  echo "{\\"step\\": $s, \\"timing_s/step\\": 1.0}" >> "$d/metrics.jsonl"
done
"""


def _sandbox(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "slurm").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    shutil.copytree(REPO / "configs", repo / "configs")
    shutil.copytree(REPO / "kaggle", repo / "kaggle")
    (repo / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (repo / "slurm" / "run_train.sh").write_text(TRAINER)
    (repo / "scripts" / "hub_sync.py").write_text(
        "import os, sys\nopen(os.environ['HUB_LOG'], 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
    )
    (repo / "scripts" / "compose_config.py").write_text("import sys\nsys.exit(1)\n")
    return repo


def _run(repo: Path, tmp_path: Path, hours: float, total: int = 5) -> subprocess.CompletedProcess:
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": os.environ["PATH"],
        "USER": "u",
        "MC_STAGE_ROOT": str(tmp_path / "stage"),
        "MC_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "UV_PYTHON_INSTALL_DIR": str(tmp_path / "uvpy"),
        "MC_OUTPUT_DIR": str(tmp_path / "out"),
        "MC_HUB_REPO": "u/r",
        "HUB_LOG": str(tmp_path / "hub.log"),
        "MC_TOTAL_STEPS": str(total),
        "MC_SESSION_START_EPOCH": str(int(time.time())),
        "MC_SESSION_HOURS": str(hours),
        "MC_SESSION_MARGIN_MIN": "0",
        "MC_POLL_SECS": "1",
    }
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "kaggle/train.sh", "math_mixed_cuts", "1", "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_stops_before_the_limit_then_resumes_and_finishes(tmp_path: Path):
    repo = _sandbox(tmp_path)
    tracker = tmp_path / "stage/runs/math_mixed_cuts-t4-s1/checkpoints/latest_checkpointed_iteration.txt"
    first = _run(repo, tmp_path, hours=4.5 / 3600)  # a 4.5-second "session"
    assert first.returncode == 0, first.stdout + first.stderr
    stopped_at = int(tracker.read_text())
    assert 1 <= stopped_at < 5, first.stdout
    assert "stopping after step" in first.stdout and "Run the notebook again" in first.stdout
    calls = (tmp_path / "hub.log").read_text().splitlines()
    assert calls[0] == "pull --run math_mixed_cuts-t4-s1"
    assert calls.count("push --run math_mixed_cuts-t4-s1") >= stopped_at, "every checkpoint is pushed"
    assert (tmp_path / "out/math_mixed_cuts-t4-s1/metrics.jsonl").exists()

    second = _run(repo, tmp_path, hours=1.0)
    assert second.returncode == 0, second.stdout + second.stderr
    assert f"resuming after step {stopped_at} of 5" in second.stdout
    assert "finished: step 5" in second.stdout and tracker.read_text().strip() == "5"
    third = _run(repo, tmp_path, hours=1.0)
    assert "already finished" in third.stdout
