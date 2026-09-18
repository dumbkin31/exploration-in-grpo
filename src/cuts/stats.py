"""Persisting and summarising per-rollout |S_t| statistics.

The logits processor lives inside vLLM's engine process (one per rollout replica), far away
from the verl trainer that logs metrics. The simplest channel that survives Ray, `mp`
executors and node-local file systems is an append-only JSONL file per engine process:

    {stats_dir}/cuts_stats_pid{pid}.jsonl

Where to write (``stats_dir``) and which training step a request belongs to travel *inside*
the request's ``extra_args["cuts"]`` (see :class:`cuts.config.CutsParams`), so no environment
variable has to cross a process boundary. The trainer reads the directory back at the end of
the step (:func:`read_cuts_stats`) and turns it into metrics (:func:`summarize_cuts_stats`).

One record per finished CUTS request; see :meth:`cuts.state.RequestEntry.summary` for fields.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

STATS_FILE_GLOB = "cuts_stats_*.jsonl"


class CutsStatsWriter:
    """Append-only JSONL writer, one open file per ``stats_dir`` seen, keyed by this PID."""

    def __init__(self) -> None:
        self._files: dict[str, IO[str]] = {}
        self._pid = os.getpid()

    def _file_for(self, stats_dir: str) -> IO[str]:
        f = self._files.get(stats_dir)
        if f is None:
            Path(stats_dir).mkdir(parents=True, exist_ok=True)
            path = Path(stats_dir) / f"cuts_stats_pid{self._pid}.jsonl"
            f = open(path, "a", encoding="utf-8")  # noqa: SIM115 - long-lived handle on purpose
            self._files[stats_dir] = f
        return f

    def write(self, record: dict[str, Any]) -> None:
        """Callback for :class:`cuts.state.CutsBatchState`; drops records without ``stats_dir``."""
        stats_dir = record.pop("stats_dir", None)
        if not stats_dir:
            return
        f = self._file_for(stats_dir)
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


def read_cuts_stats(stats_dir: str | Path, step: int | None = None) -> list[dict[str, Any]]:
    """Read every record under ``stats_dir`` (all engine processes), optionally for one step.

    Tolerates a truncated last line (an engine may be mid-write) and a missing directory.
    """
    root = Path(stats_dir)
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob(STATS_FILE_GLOB)):
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

    - ``set_size_mean``          mean |S_t| over all CUTS steps (K means "filter did nothing",
                                 1 means CUTS has degenerated into greedy decoding)
    - ``frac_steps_singleton``   fraction of CUTS steps with |S_t| == 1
    - ``frac_steps_fallback``    fraction of CUTS steps where the filter emptied the set
    - ``cuts_steps_per_rollout`` mean number of CUTS-active steps per CUTS rollout
    - ``frac_rollouts_never_active`` CUTS rollouts shorter than T_warm (CUTS never fired)
    - ``n_rollouts``             number of CUTS rollouts summarised
    """
    n_rollouts = 0
    total_steps = 0
    sum_set_size = 0.0
    n_singleton = 0
    n_fallback = 0
    never_active = 0
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
    out: dict[str, float] = {f"{prefix}n_rollouts": float(n_rollouts)}
    if n_rollouts:
        out[f"{prefix}frac_rollouts_never_active"] = never_active / n_rollouts
        out[f"{prefix}cuts_steps_per_rollout"] = total_steps / n_rollouts
    if total_steps:
        out[f"{prefix}set_size_mean"] = sum_set_size / total_steps
        out[f"{prefix}frac_steps_singleton"] = n_singleton / total_steps
        out[f"{prefix}frac_steps_fallback"] = n_fallback / total_steps
    return out
