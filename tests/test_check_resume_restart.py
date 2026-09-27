"""check_resume must find the restart both when steps go backwards and when the resume is seamless."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("check_resume", REPO / "scripts" / "check_resume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_backwards_step_is_the_restart(tmp_path: Path):
    steps = [1, 2, 3, 4, 5, 6, 7, 6, 7, 8]  # killed after 7, checkpoint at 5 -> resumed at 6
    assert _mod().find_restart(steps, [0.0] * len(steps), tmp_path / "jobs") == 7


def test_seamless_resume_uses_the_second_jobs_start_time(tmp_path: Path):
    steps = list(range(1, 31))
    times = [1000.0 + 10 * i for i in range(15)] + [5000.0 + 10 * i for i in range(15)]
    for job, t in (("100", 900.0), ("101", 4990.0)):
        d = tmp_path / "jobs" / job
        d.mkdir(parents=True)
        (d / "job_info.txt").write_text(f"job_id={job}\ndate={datetime.fromtimestamp(t):%Y-%m-%d %H:%M:%S}\n")
    assert _mod().find_restart(steps, times, tmp_path / "jobs") == 15  # row of step 16


def test_no_second_job_means_no_restart(tmp_path: Path):
    steps = list(range(1, 31))
    assert _mod().find_restart(steps, [float(i) for i in steps], tmp_path / "jobs") is None
