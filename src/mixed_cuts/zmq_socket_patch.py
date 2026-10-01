"""Give each concurrent run its own vLLM weight-transfer socket (decision 013, first H200 run).

verl v0.9.0 sends merged weights from the trainer to the colocated vLLM worker over a ZMQ IPC socket
named ``/tmp/rl-colocate-zmq-<ray job id>-replica-<r>-rank-<l>.sock``: the sender in
``vllm_rollout.ServerAdapter.__init__`` (vllm_rollout.py:141), the receiver in
``utils.vLLMColocateWorkerExtension._get_zmq_handle`` (utils.py:415), which reads the job id from
``VERL_RAY_JOB_ID``, set by ``vllm_async_server.vLLMHttpServer.__init__`` (vllm_async_server.py:133)
before the vLLM workers are spawned. The job id was meant to keep concurrent jobs apart, but
jarvis/train.sh gives every run its own Ray instance, whose first job is always ``01000000``: two arms
on one 2-GPU machine bound the same socket, and the GRPO arm's vLLM waited forever for weights the
Mixed-CUTS arm had taken (2026-10-01).

This patch inserts the run name (``MC_RUN_NAME``, exported by slurm/common.sh before the trainer
starts, inherited by every Ray worker) into both sides: the sender's path, and ``VERL_RAY_JOB_ID``
right after the server sets it, so the vLLM workers it spawns later compute the same path. Without
``MC_RUN_NAME`` (GPU tests, other tools) nothing changes. Installed lazily by
``mixed_cuts.worker_hooks.install`` like the LoRA-sync patch.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import os
import re
import sys
from collections.abc import Callable
from typing import Any

SENDER = "verl.workers.rollout.vllm_rollout.vllm_rollout"
SERVER = "verl.workers.rollout.vllm_rollout.vllm_async_server"
PREFIX = "rl-colocate-zmq-"
_FLAG = "_mixed_cuts_zmq_tag"


def run_tag() -> str:
    """The run name made safe for a socket path; '' when unset. Unix socket paths are limited to 107
    bytes and the rest of verl's path takes ~55, so the tag is cut at 40 characters."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.environ.get("MC_RUN_NAME", ""))[:40]


def tag_handle(handle: str, tag: str) -> str:
    return handle.replace(PREFIX, f"{PREFIX}{tag}-", 1) if tag and PREFIX in handle else handle


def _wrap_init(cls: Any, after: Callable[[Any], None], what: str) -> bool:
    init = cls.__init__
    if getattr(init, _FLAG, False):
        return False

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        after(self)

    setattr(__init__, _FLAG, True)
    __init__.__wrapped__ = init
    cls.__init__ = __init__
    print(f"[mixed_cuts] zmq_socket_patch installed in pid {os.getpid()}: {what}", flush=True)
    return True


def apply_sender(module: Any) -> bool:
    cls = getattr(module, "ServerAdapter", None)
    if cls is None:
        return False

    def after(self) -> None:
        tag = run_tag()
        if isinstance(getattr(self, "zmq_handle", None), str):
            self.zmq_handle = tag_handle(self.zmq_handle, tag)

    return _wrap_init(cls, after, "weight-sync sender socket carries the run name")


def apply_server(module: Any) -> bool:
    cls = getattr(module, "vLLMHttpServer", None)
    if cls is None:
        return False

    def after(self) -> None:
        tag = run_tag()
        job = os.environ.get("VERL_RAY_JOB_ID", "0")
        if tag and not job.startswith(f"{tag}-"):
            os.environ["VERL_RAY_JOB_ID"] = f"{tag}-{job}"  # read by the vLLM workers spawned later

    return _wrap_init(cls, after, "vLLM workers' receiver socket carries the run name")


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    """One-shot finder: lets the regular finders load ``target``, then runs ``fn`` on it."""

    def __init__(self, target: str, fn: Callable[[Any], bool]):
        self.target, self.fn = target, fn

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002 - importlib's signature
        if fullname != self.target:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
                break
        else:
            return None
        loader, orig_exec, fn = spec.loader, spec.loader.exec_module, self.fn

        def exec_module(module):
            orig_exec(module)
            fn(module)

        loader.exec_module = exec_module
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self)
        return spec


def _install_one(target: str, fn: Callable[[Any], bool]) -> None:
    module = sys.modules.get(target)
    if module is not None:
        fn(module)
    elif not any(isinstance(f, _PatchAfterImport) and f.target == target for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport(target, fn))


def install(sender: str = SENDER, server: str = SERVER) -> None:
    if os.environ.get("MC_ZMQ_SOCKET_PATCH", "1") == "0":
        return
    _install_one(sender, apply_sender)
    _install_one(server, apply_server)
