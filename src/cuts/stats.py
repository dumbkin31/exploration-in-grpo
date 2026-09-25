"""Persisting and summarising per-rollout |S_t| statistics.

The logits processor lives inside vLLM's engine process (with tensor parallelism: one process per
TP rank), far away from the verl trainer that logs metrics. The channel that survives Ray, `mp`
executors and node-local file systems is an append-only JSONL file per (training step, process):

    {stats_dir}/cuts_stats_step{N}_{writer-id}.jsonl      (writer-id = pid + random suffix)

Design notes (see docs/decisions/007-cuts-stats-channel.md):

* ``stats_dir`` and the training ``step`` travel *inside* the request's ``extra_args["cuts"]``
  (:class:`cuts.config.CutsParams`), so nothing crosses a process boundary through env vars.
* With TP > 1 every rank runs the sampler and the processor on the full batch; only TP rank 0
  owns a writer (:mod:`cuts.vllm_logits_processor`), so each rollout yields exactly one record.
* A request's record is emitted when vLLM removes it from the persistent batch, which happens at
  the engine's *next* forward pass. The last rollouts of a generation batch are therefore flushed
  when the next batch starts, and the trainer summarises step N when it reaches step N+1.
* Files are keyed by step so the trainer reads O(one step) per step, and a restarted run can
  ignore files that pre-date it (``list_stats_files`` + ``ignore``).

One record per finished CUTS request; see :meth:`cuts.state.RequestEntry.summary` for fields.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

STATS_FILE_GLOB = "cuts_stats_*.jsonl"


def stats_file_name(step: int | None, writer_id: str) -> str:
    return f"cuts_stats_step{'NA' if step is None else int(step)}_{writer_id}.jsonl"


def stats_file_glob(step: int | None) -> str:
    return f"cuts_stats_step{'NA' if step is None else int(step)}_*.jsonl"


class CutsStatsWriter:
    """Append-only JSONL writer; one open file per (stats_dir, step) seen, keyed by this PID."""

    def __init__(self, tp_rank: int = 0) -> None:
        self._files: dict[tuple[str, int | None], IO[str]] = {}
        self._pid = os.getpid()
        self._tp_rank = int(tp_rank)
        # PIDs repeat across nodes and across restarts of a run that share one stats_dir on
        # /share1, so every writer instance gets its own file name suffix.
        self.writer_id = f"pid{self._pid}-{uuid.uuid4().hex[:8]}"

    def _file_for(self, stats_dir: str, step: int | None) -> IO[str]:
        key = (stats_dir, step)
        f = self._files.get(key)
        if f is None:
            # Keep at most a few files open: a new step means the previous ones are finished.
            for old_key in [k for k in self._files if k[0] == stats_dir and k[1] != step]:
                try:
                    self._files.pop(old_key).close()
                except OSError:
                    pass
            Path(stats_dir).mkdir(parents=True, exist_ok=True)
            f = open(Path(stats_dir) / stats_file_name(step, self.writer_id), "a", encoding="utf-8")  # noqa: SIM115
            self._files[key] = f
        return f

    def write(self, record: dict[str, Any]) -> None:
        """Callback for :class:`cuts.state.CutsBatchState`; drops records without ``stats_dir``."""
        stats_dir = record.pop("stats_dir", None)
        if not stats_dir:
            return
        record["tp_rank"] = self._tp_rank
        record["pid"] = self._pid
        f = self._file_for(str(stats_dir), record.get("step"))
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()  # a crashed engine must not lose the step's statistics

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except OSError:
                pass
        self._files.clear()

    def __del__(self) -> None:  # best effort
        self.close()


def list_stats_files(stats_dir: str | Path) -> set[str]:
    """Names of every stats file currently under ``stats_dir`` (for snapshotting at trainer init)."""
    root = Path(stats_dir)
    if not root.is_dir():
        return set()
    return {p.name for p in root.glob(STATS_FILE_GLOB)}


def read_cuts_stats(
    stats_dir: str | Path,
    step: int | None = None,
    ignore: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Read records under ``stats_dir``: one step's files when ``step`` is given, else all of them.

    ``ignore`` is a set of file names to skip (files left by a previous incarnation of the run).
    Tolerates a truncated last line (an engine may be mid-write) and a missing directory.
    """
    root = Path(stats_dir)
    if not root.is_dir():
        return []
    skip = set(ignore or ())
    pattern = STATS_FILE_GLOB if step is None else stats_file_glob(step)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        if path.name in skip:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line
                if step is None or rec.get("step") == step:
                    records.append(rec)
    return records


def summarize_cuts_stats(records: Iterable[dict[str, Any]], prefix: str = "cuts/") -> dict[str, float]:
    """Aggregate per-request records into the metrics logged every training step.

    Weighted by decoding steps, so a long rollout counts as many CUTS steps:

    - ``set_size_mean``              mean |S_t| over all CUTS steps (K = "filter did nothing",
                                     1 = CUTS has degenerated into greedy decoding)
    - ``set_size_hist/k=i``          fraction of CUTS steps with |S_t| == i (i = 1..K)
    - ``degenerate_to_greedy_rate``  fraction of CUTS steps with |S_t| == 1 (alias:
                                     ``frac_steps_singleton``)
    - ``frac_steps_fallback``        fraction of CUTS steps where the filter emptied the set
    - ``cuts_steps_per_rollout``     mean number of CUTS-active steps per CUTS rollout
    - ``frac_rollouts_never_active`` CUTS rollouts shorter than T_warm (CUTS never fired)
    - ``n_rollouts``                 number of CUTS rollouts summarised
    """
    n_rollouts = 0
    total_steps = 0
    sum_set_size = 0.0
    n_singleton = 0
    n_fallback = 0
    never_active = 0
    hist: list[int] = []
    for rec in records:
        n_rollouts += 1
        steps = int(rec.get("n_cuts_steps") or 0)
        if steps == 0:
            never_active += 1
            continue
        total_steps += steps
        sum_set_size += float(rec.get("mean_set_size") or 0.0) * steps
        n_singleton += int(rec.get("n_singleton") or 0)
        n_fallback += int(rec.get("n_fallback") or 0)
        h = rec.get("set_size_hist") or []
        if len(h) > len(hist):
            hist.extend([0] * (len(h) - len(hist)))
        for i, c in enumerate(h):
            hist[i] += int(c)
    out: dict[str, float] = {f"{prefix}n_rollouts": float(n_rollouts)}
    if n_rollouts:
        out[f"{prefix}frac_rollouts_never_active"] = never_active / n_rollouts
        out[f"{prefix}cuts_steps_per_rollout"] = total_steps / n_rollouts
    if total_steps:
        out[f"{prefix}set_size_mean"] = sum_set_size / total_steps
        out[f"{prefix}frac_steps_singleton"] = n_singleton / total_steps
        out[f"{prefix}degenerate_to_greedy_rate"] = n_singleton / total_steps
        out[f"{prefix}frac_steps_fallback"] = n_fallback / total_steps
        hist_total = sum(hist)
        if hist_total:
            for i, c in enumerate(hist, start=1):
                out[f"{prefix}set_size_hist/k={i}"] = c / hist_total
    return out
