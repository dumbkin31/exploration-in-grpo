"""Length-sorted dynamic micro-batches on the padded attention path (decision 015).

Without the flash-attn package the actor runs with ``use_remove_padding: false`` (configs/train/hardware),
so ``FSDPEngine.prepare_model_inputs`` pads every micro-batch to its longest sequence
(verl/workers/engine/fsdp/transformer_impl.py:1259). verl v0.9.0 forms the micro-batches in
``seqlen_balancing.rearrange_micro_batches``, which balances the attention workload between them
(Karmarkar-Karp on ``24576 * len + len**2``), so every micro-batch mixes long and short sequences. With
GRPO's length spread (mean ~1,030 tokens, a long tail to 5,112) only ~25% of the padded tokens are
real: Mixed-CUTS step 1 on the H200 spent 950 of its 1,066 seconds in the three training passes
(old log-prob, ref, update) at an MFU of 0.06 (2026-10-01).

This patch keeps verl's call and replaces only the partition: sequences are sorted by length and cut into
runs that fit the token budget once padded (``count * longest <= max_token_len``), longest run first so
the memory peak comes at once. Padding efficiency on the measured lengths goes from ~0.25 to ~0.97, and
each micro-batch's padded size, hence its activation and logits memory, stays below what verl gave it.

The gradient does not change: ``FSDPEngine.forward_backward_batch`` counts the loss tokens over the whole
mini-batch before splitting it (``batch_num_tokens``, transformer_impl.py:706) and ``agg_loss`` divides
each micro-batch's token sum by that count, so the partition only changes the floating-point summation
order. The outputs are put back in order by verl's ``restore_dynamic_batch`` from the index lists
returned here.

The patch steps aside (calls verl's function unchanged) whenever its assumptions do not hold: the
remove-padding path (no padding to save), more than one data-parallel rank (ranks must agree on the
micro-batch count), grouped samples, pipeline divisors or a forced minimum count. Set
``MC_SORTED_MICROBATCHES=0`` to keep verl's behaviour. Installed lazily through
``mixed_cuts.worker_hooks.install`` like the other patches.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

TARGET = "verl.workers.engine.utils"
_FLAG = "_mixed_cuts_sorted_microbatches"


def sorted_partitions(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Index lists of similar-length sequences, each fitting ``budget`` tokens once padded to its longest.

    A sequence longer than the budget gets a micro-batch of its own (the wrapper hands that case back to
    verl, which asserts). Longest micro-batch first; indices inside a micro-batch ascend by length.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    parts: list[list[int]] = []
    cur: list[int] = []
    for i in order:  # ascending, so lengths[i] is the longest of cur + [i]
        if cur and lengths[i] * (len(cur) + 1) > budget:
            parts.append(cur)
            cur = []
        cur.append(i)
    if cur:
        parts.append(cur)
    parts.reverse()
    return parts


def padding_efficiency(lengths: Sequence[int], parts: Sequence[Sequence[int]]) -> float:
    padded = sum(len(p) * max(lengths[i] for i in p) for p in parts if p)
    return sum(lengths[i] for p in parts for i in p) / padded if padded else 1.0


def _single_rank(dp_group: Any) -> bool:
    if dp_group is None:
        return True
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_world_size(dp_group) == 1


def steps_aside(args: dict[str, Any], use_remove_padding: bool) -> str | None:
    """Why verl's own partition must be used for this call, or None when the patch applies."""
    if use_remove_padding:
        return "remove-padding path"
    if args.get("force_group_size", 1) != 1:
        return "grouped samples"
    if args.get("num_batches_divided_by") is not None or args.get("min_num_micro_batch") is not None:
        return "fixed micro-batch count"
    if not _single_rank(args.get("dp_group")):
        return "more than one data-parallel rank"
    return None


def _lengths(batch: Any) -> list[int]:
    input_ids = batch["input_ids"]
    if getattr(input_ids, "is_nested", False):
        return [int(x) for x in input_ids.offsets().diff().tolist()]
    return [int(x) for x in batch["attention_mask"].sum(dim=1).tolist()]


def apply(module: Any) -> bool:
    """Wrap ``module.rearrange_micro_batches``. Idempotent; True if patched."""
    orig = getattr(module, "rearrange_micro_batches", None)
    if orig is None:
        print(f"[mixed_cuts] microbatch_patch: {module.__name__} has no rearrange_micro_batches", flush=True)
        return False
    if getattr(orig, _FLAG, False):
        return True
    sig = inspect.signature(orig)
    reported: set[str] = set()

    def rearrange_micro_batches(*args, **kwargs):
        from verl.utils import tensordict_utils as tu

        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        batch, budget = a["batch"], a["max_token_len"]
        rmpad = bool(tu.get_non_tensor_data(data=batch, key="use_remove_padding", default=True))
        why = steps_aside(a, rmpad)
        lengths = None if why else _lengths(batch)
        if lengths is not None and max(lengths) > budget:
            why = "a sequence exceeds the budget"
        if why:
            if why not in reported:
                reported.add(why)
                print(f"[mixed_cuts] microbatch_patch: verl's partition kept ({why})", flush=True)
            return orig(*args, **kwargs)
        parts = sorted_partitions(lengths, budget)
        if "first" not in reported:
            reported.add("first")
            print(
                f"[mixed_cuts] microbatch_patch: {len(lengths)} sequences -> {len(parts)} length-sorted "
                f"micro-batches, padding efficiency {padding_efficiency(lengths, parts):.2f} (budget {budget})",
                flush=True,
            )
        return [tu.index_select_tensor_dict(batch, p) for p in parts], parts

    setattr(rearrange_micro_batches, _FLAG, True)
    rearrange_micro_batches.__wrapped__ = orig
    module.rearrange_micro_batches = rearrange_micro_batches
    print(
        f"[mixed_cuts] microbatch_patch installed in pid {os.getpid()}: padded micro-batches are length-sorted",
        flush=True,
    )
    return True


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


def install(target: str = TARGET) -> None:
    if os.environ.get("MC_SORTED_MICROBATCHES", "1") == "0":
        return
    module = sys.modules.get(target)
    if module is not None:
        apply(module)
    elif not any(isinstance(f, _PatchAfterImport) and f.target == target for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport(target, apply))
