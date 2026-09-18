"""Per-request bookkeeping for CUTS inside a vLLM batch.

vLLM's V1 sampler keeps a *persistent batch*: every in-flight request owns a row index in the
logits tensor, and rows are added, removed, and moved (condensed / swapped) between steps.
Before each forward pass the engine hands every logits processor a ``BatchUpdate`` with three
lists that must be applied **in this order**: ``removed`` (row indices), ``added``
(``(index, SamplingParams, prompt_tok_ids, output_tok_ids)`` tuples) and ``moved``
(``(from_index, to_index, MoveDirectionality)`` tuples, where the directionality is either
UNIDIRECTIONAL, "from" moves into "to", or SWAP). Added/moved requests may replace whatever
previously lived at the target index.

Two facts make the implementation simple:

* ``output_tok_ids`` is a *live reference* to the request's growing output list, so
  prefix protection (T_warm) is just ``len(output_tok_ids) >= t_warm``; we never count tokens
  ourselves and cannot drift from the engine.
* A request is a CUTS request iff its ``SamplingParams.extra_args`` carries the ``"cuts"``
  key; standard requests are simply never entered in the table, so the same batch can mix
  both kinds and standard rows are never touched.

This module mirrors those semantics without importing vLLM, so it can be unit-tested on a
CPU with hand-built ``BatchUpdate``-shaped objects (see ``tests/test_cuts_state.py``). The
real vLLM ``LogitsProcessor`` in :mod:`cuts.vllm_logits_processor` is a thin wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from cuts.config import CutsParams
from cuts.operator import cuts_transform


def _is_swap(direction: Any) -> bool:
    """vLLM passes its own ``MoveDirectionality`` enum; compare by name to stay import-free."""
    return str(getattr(direction, "name", direction)).upper().endswith("SWAP")


@dataclass
class RequestEntry:
    """State of one in-flight CUTS request (one row of the persistent batch)."""

    params: CutsParams
    output_tok_ids: list[int]
    """Live reference owned by vLLM; grows by one token per decoding step."""
    prompt_len: int = 0
    # --- |S_t| statistics, accumulated over the CUTS-active steps of this request ---
    n_cuts_steps: int = 0
    sum_set_size: int = 0
    n_singleton: int = 0
    """Steps where |S_t| == 1, i.e. CUTS degenerated into greedy decoding."""
    n_fallback: int = 0
    """Steps where the filter emptied the set and ``empty_set_fallback`` fired."""

    @property
    def n_generated(self) -> int:
        return len(self.output_tok_ids)

    def is_active(self) -> bool:
        """PREFIX PROTECTION: CUTS applies only once ``t_warm`` tokens have been generated."""
        return self.n_generated >= self.params.t_warm

    def record(self, set_size: int, used_fallback: bool) -> None:
        self.n_cuts_steps += 1
        self.sum_set_size += int(set_size)
        if set_size == 1:
            self.n_singleton += 1
        if used_fallback:
            self.n_fallback += 1

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable record, emitted when the request leaves the batch."""
        p = self.params
        return {
            "uid": p.uid,
            "session_id": p.session_id,
            "step": p.step,
            "k": p.k,
            "delta": p.delta,
            "t_warm": p.t_warm,
            "empty_set_fallback": p.empty_set_fallback,
            "prompt_len": self.prompt_len,
            "n_generated": self.n_generated,
            "n_cuts_steps": self.n_cuts_steps,
            "mean_set_size": (self.sum_set_size / self.n_cuts_steps) if self.n_cuts_steps else None,
            "n_singleton": self.n_singleton,
            "n_fallback": self.n_fallback,
        }


class CutsBatchState:
    """Table of in-flight CUTS requests keyed by persistent-batch row index.

    Args:
        on_request_finished: optional callback receiving :meth:`RequestEntry.summary` whenever a
            CUTS request leaves the batch (finished, aborted, or evicted). Used to persist |S_t|
            statistics; ``None`` discards them.
    """

    def __init__(self, on_request_finished: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._entries: dict[int, RequestEntry] = {}
        self._on_finished = on_request_finished

    # ------------------------------------------------------------------ introspection
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, index: int) -> bool:
        return index in self._entries

    def get(self, index: int) -> RequestEntry | None:
        return self._entries.get(index)

    def active_indices(self) -> list[int]:
        return sorted(i for i, e in self._entries.items() if e.is_active())

    # ------------------------------------------------------------------ batch updates
    def apply_batch_update(self, batch_update: Any) -> None:
        """Mirror a vLLM ``BatchUpdate``. Order matters: removed, then added, then moved."""
        if batch_update is None:
            return
        for index in batch_update.removed:
            self._finish(index)
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            # An added request may replace the previous occupant of this row.
            self._finish(index)
            cuts_params = CutsParams.from_extra_args(getattr(params, "extra_args", None))
            if cuts_params is None:
                continue  # standard request: never tracked, never touched
            self._entries[index] = RequestEntry(
                params=cuts_params,
                output_tok_ids=output_tok_ids,
                prompt_len=len(prompt_tok_ids) if prompt_tok_ids is not None else 0,
            )
        for from_index, to_index, direction in batch_update.moved:
            entry_from = self._entries.pop(from_index, None)
            entry_to = self._entries.pop(to_index, None)
            if _is_swap(direction):
                if entry_from is not None:
                    self._entries[to_index] = entry_from
                if entry_to is not None:
                    self._entries[from_index] = entry_to
            else:
                # UNIDIRECTIONAL: "from" moves into "to". vLLM only moves into a vacated row, so
                # entry_to is normally None; if it is not, finish it rather than leak it.
                if entry_to is not None:
                    self._emit(entry_to)
                if entry_from is not None:
                    self._entries[to_index] = entry_from

    def _finish(self, index: int) -> RequestEntry | None:
        entry = self._entries.pop(index, None)
        if entry is not None:
            self._emit(entry)
        return entry

    def _emit(self, entry: RequestEntry) -> None:
        if self._on_finished is not None:
            self._on_finished(entry.summary())

    def clear(self) -> None:
        """Finish every tracked request (engine shutdown)."""
        for index in list(self._entries):
            self._finish(index)

    # ------------------------------------------------------------------ the operator
    def apply(self, logits: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        """Apply CUTS to every active row of ``logits`` and record |S_t| statistics.

        Requests may carry different ``(k, delta, fallback)`` settings, so rows are grouped by
        :attr:`CutsParams.signature` and the operator runs once per distinct setting (in the
        common case, once). Rows without an active CUTS request are left untouched.
        """
        batch = logits.shape[0]
        groups: dict[tuple[int, float, str], list[int]] = {}
        for index, entry in self._entries.items():
            if index < batch and entry.is_active():
                groups.setdefault(entry.params.signature, []).append(index)
        if not groups:
            return logits

        out = logits
        for (k, delta, fallback), indices in groups.items():
            mask = torch.zeros(batch, dtype=torch.bool, device=logits.device)
            mask[indices] = True
            result = cuts_transform(
                out, k=k, delta=delta, active_mask=mask, empty_set_fallback=fallback, inplace=inplace
            )
            out = result.logits
            inplace = True  # later groups can safely overwrite the tensor we now own
            # One small device->host copy per distinct setting per step.
            stats = torch.stack([result.set_size, result.used_fallback.to(torch.int64)]).cpu()
            for i in indices:
                self._entries[i].record(int(stats[0, i]), bool(stats[1, i]))
        return out
