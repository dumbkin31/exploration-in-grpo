"""Sync merged LoRA weights to vLLM without verl's 6.9 GB CPU backup of the base model.

verl v0.9.0 streams merged (base + LoRA) weights to the rollout inside
``merged_lora_context(self.module, backup_adapters=True)``
(verl/workers/engine/fsdp/transformer_impl.py:1048, ``FSDPEngine._merged_lora_per_tensor_param``; the
only caller of the context in verl). ``backup_adapters=True`` first runs ``backup_base_model_weights``,
which clones every non-LoRA parameter into a fresh CPU tensor: 1.72 B fp32 parameters, 6.9 GB, allocated
at once right after the checkpoint save. On the research QoS (30 GB host cgroup) the first arm sat at
~27.5 GB when the weight sync began and was OOM-killed inside that clone at the end of step 1 (jobs
2719894, 2719898, 2721833; decision 012). There is no config switch for it.

With ``backup_adapters=False`` verl's own fallback path runs instead: after the weights are streamed,
``fsdp_merge_unmerge(module, do_merge=False)`` calls PEFT's ``LoraLayer.unmerge`` layer by layer, which
subtracts the same ``scaling * B @ A`` that ``merge`` added. The frozen fp32 base weights then carry at
most ~1 ulp of rounding per step (relative 1e-7; tests/test_lora_sync_patch.py measures the round trip),
far below the fp16 precision the rollout samples with. After every sync the wrapper checks that no LoRA
layer is still merged and raises otherwise (a merged layer would double-count the adapter in training).

The patch touches only ``transformer_impl``'s reference to the context (``collect_merged_lora_params``,
the other user of the backup, has no callers in verl 0.9.0). It installs lazily: most Ray workers never
import the FSDP engine, and importing it in all of them would cost host memory on a 30 GB budget. Set
``MC_LORA_SYNC_BACKUP=1`` to keep verl's behaviour. Installed through ``mixed_cuts.worker_hooks.install``.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import os
import sys
from typing import Any

TARGET = "verl.workers.engine.fsdp.transformer_impl"
_FLAG = "_mixed_cuts_lora_sync_no_backup"


def _any_merged_lora(actor: Any) -> bool:
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:  # no PEFT: nothing can be merged
        return False
    return any(isinstance(m, LoraLayer) and getattr(m, "merged", False) for m in actor.modules())


def apply(module: Any) -> bool:
    """Replace ``module.merged_lora_context`` with the no-backup wrapper. Idempotent; True if patched."""
    orig = getattr(module, "merged_lora_context", None)
    if orig is None:
        print(
            f"[mixed_cuts] lora_sync_patch: {module.__name__} has no merged_lora_context; not patched",
            flush=True,
        )
        return False
    if getattr(orig, _FLAG, False):
        return True

    @contextlib.contextmanager
    def merged_lora_context(actor, backup_adapters=False):  # noqa: ARG001 - verl's signature; always False
        with orig(actor, backup_adapters=False):
            yield
        if _any_merged_lora(actor):
            raise RuntimeError("LoRA layers still merged after the weight sync: the unmerge did not run")

    setattr(merged_lora_context, _FLAG, True)
    module.merged_lora_context = merged_lora_context
    print(
        f"[mixed_cuts] lora_sync_patch installed in pid {os.getpid()}: merged-LoRA weight sync unmerges "
        "instead of backing up the base weights to CPU",
        flush=True,
    )
    return True


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    """One-shot finder: lets the regular finders load ``target``, then patches it right after exec."""

    def __init__(self, target: str):
        self.target = target

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
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            apply(module)

        loader.exec_module = exec_module  # the loader instance belongs to this spec only
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self)
        return spec


def install(target: str = TARGET) -> None:
    if os.environ.get("MC_LORA_SYNC_BACKUP", "0") == "1":
        return
    module = sys.modules.get(target)
    if module is not None:
        apply(module)
        return
    if not any(isinstance(f, _PatchAfterImport) and f.target == target for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport(target))
