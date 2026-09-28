"""slurm/common.sh: a job that lands on a bad node records it and resubmits itself elsewhere.

Runs the real bash functions against fake sbatch/scontrol/squeue/scancel that log their arguments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKES = {
    "sbatch": 'echo "sbatch $*" >> "$FAKE_LOG"; [ -n "${FAKE_SBATCH_FAIL:-}" ] && exit 1; echo "999;ada"\n',
    "scontrol": """echo "scontrol $*" >> "$FAKE_LOG"
case "$1" in
  show) echo "JobId=123 JobName=math_grpo-s1 UserId=u(1) Account=research QOS=low TimeLimit=${FAKE_TLIMIT:-4-00:00:00} ReqNodeList=${FAKE_REQ:-(null)} ExcNodeList=gnode043 Command=${FAKE_CMD} WorkDir=/w StdOut=/w/o" ;;
  update) [ -n "${FAKE_UPDATE_FAIL:-}" ] && exit 1 ;;
esac
exit 0
""",
    "squeue": 'echo "squeue $*" >> "$FAKE_LOG"; printf "%s" "${FAKE_PENDING:-}" | tr "," "\\n"\n',
    "scancel": 'echo "scancel $*" >> "$FAKE_LOG"\n',
}


def _run(tmp_path: Path, call: str, **env: str) -> tuple[int, list[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in FAKES.items():
        f = bin_dir / name
        f.write_text("#!/usr/bin/env bash\n" + body)
        f.chmod(0o755)
    script = tmp_path / "train.sbatch"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "calls.log"
    log.write_text("")
    driver = f"""
mc_sbatch_args() {{ printf -- '-A research --qos=low -c 10 -x gnode043,gnode047'; }}
mc_sbatch_exclude() {{ printf -- '-x gnode043,gnode047'; }}
source "{REPO}/slurm/common.sh"
set +e
{call}
echo "RC=$?"
"""
    full_env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "USER": "u",
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_NAME": "math_grpo-s1",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "FAKE_LOG": str(log),
        "FAKE_CMD": str(script),
        "MC_BAD_NODES_FILE": str(tmp_path / "bad_nodes.txt"),
        **env,
    }
    p = subprocess.run(["bash", "-c", driver], capture_output=True, text=True, env=full_env, timeout=60)
    rc_lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RC=")]
    rc = int(rc_lines[-1][3:]) if rc_lines else p.returncode
    return rc, [ln for ln in log.read_text().splitlines() if ln], p.stdout + p.stderr


def _sbatch(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("sbatch ")]


def test_plain_job_resubmits_with_same_name_time_and_script(tmp_path: Path):
    rc, calls, out = _run(tmp_path, "mc_resubmit_elsewhere")
    assert rc == 0, out
    (sb,) = _sbatch(calls)
    assert "-J math_grpo-s1" in sb and "-t 4-00:00:00" in sb and sb.endswith(str(tmp_path / "train.sbatch"))
    assert "--export=ALL,MC_RESUBMIT_ATTEMPT=1" in sb and "-x gnode043,gnode047" in sb
    assert "--dependency" not in sb and "--hold" not in sb
    assert "auto-resubmitted as job 999 (attempt 1 of 5)" in out


def test_serial_job_without_twin_keeps_singleton(tmp_path: Path):
    rc, calls, out = _run(tmp_path, "mc_resubmit_elsewhere", MC_RESUBMIT_SERIAL="1")
    assert rc == 0, out
    (sb,) = _sbatch(calls)
    assert "--dependency=singleton" in sb and "--hold" not in sb
    assert not any(c.startswith("scontrol update") for c in calls)


def test_serial_job_with_pending_twin_goes_first(tmp_path: Path):
    rc, calls, out = _run(tmp_path, "mc_resubmit_elsewhere", MC_RESUBMIT_SERIAL="1", FAKE_PENDING="555,123")
    assert rc == 0, out
    (sb,) = _sbatch(calls)
    assert "--hold" in sb and "--dependency" not in sb
    order = [c.split()[1] for c in calls if c.startswith(("sbatch", "scontrol update", "scontrol release"))]
    assert order[0].startswith("--parsable")
    assert "scontrol update JobId=555 Dependency=afterany:999,singleton" in calls
    assert not any("JobId=123" in c for c in calls if c.startswith("scontrol update")), (
        "never re-points itself"
    )
    assert calls.index("scontrol update JobId=555 Dependency=afterany:999,singleton") < calls.index(
        "scontrol release 999"
    )


def test_serial_twin_update_failure_cancels_the_new_job(tmp_path: Path):
    rc, calls, out = _run(
        tmp_path, "mc_resubmit_elsewhere", MC_RESUBMIT_SERIAL="1", FAKE_PENDING="555", FAKE_UPDATE_FAIL="1"
    )
    assert rc == 1
    assert "scancel 999" in calls and "scontrol release 999" not in calls


@pytest.mark.parametrize(
    ("env", "expected_rc"),
    [
        ({"FAKE_REQ": "gnode084"}, 1),  # pinned with -w: its checkpoints live on that node
        ({"MC_RESUBMIT_ATTEMPT": "5"}, 1),  # cap reached
        ({"MC_AUTO_RESUBMIT": "0"}, 0),  # switched off
    ],
)
def test_no_resubmission(tmp_path: Path, env: dict, expected_rc: int):
    rc, calls, out = _run(tmp_path, "mc_resubmit_elsewhere", **env)
    assert rc == expected_rc, out
    assert not _sbatch(calls)


def test_attempt_counter_increments(tmp_path: Path):
    rc, calls, out = _run(tmp_path, "mc_resubmit_elsewhere", MC_RESUBMIT_ATTEMPT="2")
    assert rc == 0, out
    assert "MC_RESUBMIT_ATTEMPT=3" in _sbatch(calls)[0]


def test_bad_node_exit_records_resubmits_and_exits_6(tmp_path: Path):
    rc, calls, out = _run(tmp_path, '( mc_bad_node_exit "driver=570.211.01" ); echo "SUB=$?"')
    assert "SUB=6" in out
    import socket

    line = (tmp_path / "bad_nodes.txt").read_text().strip()
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip() or socket.gethostname()
    assert line.startswith(f"{host} driver=570.211.01 job=123 ")
    assert len(_sbatch(calls)) == 1


def test_bad_node_exit_still_exits_6_when_sbatch_fails(tmp_path: Path):
    rc, calls, out = _run(
        tmp_path, '( mc_bad_node_exit "cuda_init_failed gpu=0" ); echo "SUB=$?"', FAKE_SBATCH_FAIL="1"
    )
    assert "SUB=6" in out and "auto-resubmit failed" in out


def test_dry_run_prints_and_submits_nothing(tmp_path: Path):
    rc, calls, out = _run(
        tmp_path, "mc_resubmit_elsewhere", MC_RESUBMIT_SERIAL="1", FAKE_PENDING="555", MC_RESUBMIT_DRY_RUN="1"
    )
    assert rc == 0, out
    assert not _sbatch(calls) and not any(
        c.startswith(("scontrol update", "scontrol release")) for c in calls
    )
    assert "DRY RUN: sbatch --parsable" in out and "--hold" in out and "re-point pending [555" in out
