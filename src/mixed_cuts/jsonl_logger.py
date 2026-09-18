"""Local JSONL metrics log: one JSON object per training step.

This is the *primary* log (W&B is secondary and runs offline on Ada). It must never raise into
the training loop and must survive a killed job, so every record is flushed immediately.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert numpy / torch scalars and arrays to plain Python so json.dumps works."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    ndim = getattr(value, "ndim", None)
    if ndim == 0 and hasattr(value, "item"):
        return value.item()  # 0-d numpy / torch scalar
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()  # arrays / tensors of any rank, 1-element ones included
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonlMetricsLogger:
    """Append ``{"step": ..., "time": ..., **metrics}`` lines to ``path``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        record = {"step": int(step), "time": time.time(), **to_jsonable(dict(metrics))}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
