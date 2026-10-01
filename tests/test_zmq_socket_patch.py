"""zmq_socket_patch: two runs on one machine get different vLLM weight-transfer sockets."""

from __future__ import annotations

import importlib
import inspect
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from mixed_cuts import worker_hooks, zmq_socket_patch

# the two verl modules, reduced to the lines that build the socket path (vllm_rollout.py:141,
# vllm_async_server.py:133, utils.py:415)
SENDER_SRC = textwrap.dedent(
    """
    class ServerAdapter:
        def __init__(self, job_id="01000000", replica_rank=0, local_rank=0):
            self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock"
    """
)
SERVER_SRC = textwrap.dedent(
    """
    import os

    class vLLMHttpServer:
        def __init__(self, job_id="01000000"):
            os.environ["VERL_RAY_JOB_ID"] = job_id

    def receiver_handle(replica_rank=0, local_rank=0):
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock"
    """
)


@pytest.fixture
def fake_verl(tmp_path: Path, monkeypatch):
    pkg = f"mc_fake_vllm_rollout_{uuid.uuid4().hex[:8]}"
    (tmp_path / pkg).mkdir()
    (tmp_path / pkg / "__init__.py").write_text("")
    (tmp_path / pkg / "vllm_rollout.py").write_text(SENDER_SRC)
    (tmp_path / pkg / "vllm_async_server.py").write_text(SERVER_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("VERL_RAY_JOB_ID", raising=False)
    monkeypatch.delenv("MC_ZMQ_SOCKET_PATCH", raising=False)
    sender, server = f"{pkg}.vllm_rollout", f"{pkg}.vllm_async_server"
    yield sender, server
    for name in [n for n in sys.modules if n.startswith(pkg)]:
        del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if getattr(f, "target", None) not in (sender, server)]


def _handles(sender: str, server: str) -> tuple[str, str]:
    snd = importlib.import_module(sender).ServerAdapter().zmq_handle
    srv = importlib.import_module(server)
    srv.vLLMHttpServer()
    return snd, srv.receiver_handle()


def test_sender_and_receiver_agree_and_carry_the_run_name(fake_verl, monkeypatch):
    sender, server = fake_verl
    monkeypatch.setenv("MC_RUN_NAME", "math_grpo-h200-s1")
    zmq_socket_patch.install(sender, server)
    snd, rcv = _handles(sender, server)
    assert snd == rcv == "ipc:///tmp/rl-colocate-zmq-math_grpo-h200-s1-01000000-replica-0-rank-0.sock"
    assert len(snd) - len("ipc://") < 108, "unix socket path limit"


def test_two_runs_with_the_same_ray_job_id_get_different_sockets(fake_verl, monkeypatch):
    sender, server = fake_verl
    zmq_socket_patch.install(sender, server)
    monkeypatch.setenv("MC_RUN_NAME", "math_grpo-h200-s1")
    a = _handles(sender, server)
    monkeypatch.setenv("MC_RUN_NAME", "math_mixed_cuts-h200-s1")
    monkeypatch.delenv("VERL_RAY_JOB_ID", raising=False)
    b = _handles(sender, server)
    assert a[0] == a[1] and b[0] == b[1] and a[0] != b[0]


def test_without_a_run_name_nothing_changes(fake_verl, monkeypatch):
    sender, server = fake_verl
    monkeypatch.delenv("MC_RUN_NAME", raising=False)
    zmq_socket_patch.install(sender, server)
    snd, rcv = _handles(sender, server)
    assert snd == rcv == "ipc:///tmp/rl-colocate-zmq-01000000-replica-0-rank-0.sock"


def test_already_imported_modules_are_patched_once(fake_verl, monkeypatch):
    sender, server = fake_verl
    monkeypatch.setenv("MC_RUN_NAME", "r1")
    mod = importlib.import_module(sender)
    zmq_socket_patch.install(sender, server)
    init = mod.ServerAdapter.__init__
    zmq_socket_patch.install(sender, server)
    assert mod.ServerAdapter.__init__ is init
    assert "zmq-r1-01000000" in mod.ServerAdapter().zmq_handle


def test_tag_is_sanitised_and_bounded(monkeypatch):
    monkeypatch.setenv("MC_RUN_NAME", "a b/c" + "x" * 80)
    tag = zmq_socket_patch.run_tag()
    assert tag.startswith("a_b_c") and len(tag) == 40


def test_worker_hook_installs_all_three(monkeypatch):
    called = []
    monkeypatch.setattr("mixed_cuts.sdpa_patch.install", lambda: called.append("sdpa"))
    monkeypatch.setattr("mixed_cuts.lora_sync_patch.install", lambda: called.append("lora"))
    monkeypatch.setattr("mixed_cuts.zmq_socket_patch.install", lambda: called.append("zmq"))
    worker_hooks.install()
    assert called == ["sdpa", "lora", "zmq"]


def test_verl_still_builds_the_paths_this_patch_expects():
    """Guard for verl upgrades: the three places the patch relies on."""
    rollout = pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_rollout")
    server = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    utils = importlib.import_module("verl.workers.rollout.vllm_rollout.utils")
    assert "ipc:///tmp/rl-colocate-zmq-{job_id}-replica-" in inspect.getsource(rollout.ServerAdapter.__init__)
    assert 'os.environ["VERL_RAY_JOB_ID"]' in inspect.getsource(server.vLLMHttpServer.__init__)
    src = inspect.getsource(utils.vLLMColocateWorkerExtension._get_zmq_handle)
    assert 'os.environ.get("VERL_RAY_JOB_ID"' in src and "rl-colocate-zmq-{job_id}" in src
