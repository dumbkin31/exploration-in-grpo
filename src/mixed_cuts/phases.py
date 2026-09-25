"""Phase markers for the memory profiler: (step, phase, t_start, t_end) lines in ``phases.jsonl``.

``scripts/profile_memory.py --report`` joins these wall-clock windows with the nvidia-smi
samples in ``gpu_mem.jsonl`` to give peak memory per GPU for the rollout, old-log-prob,
ref-log-prob and actor-update phases.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, float] = {}

    def begin(self, step: int, phase: str) -> None:
        self._open[phase] = time.time()

    def end(self, step: int, phase: str) -> None:
        t0 = self._open.pop(phase, None)
        if t0 is None:
            return
        self.record(step, phase, t0, time.time())

    def record(self, step: int, phase: str, t_start: float, t_end: float) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"step": int(step), "phase": phase, "t_start": t_start, "t_end": t_end}) + "\n"
            )

    @contextmanager
    def phase(self, step: int, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.record(step, name, t0, time.time())

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
