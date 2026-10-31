"""Report the percentage of CUTS tokens generated from singleton sets.

Each CUTS stats record stores ``n_singleton`` (tokens generated when ``|S_t| == 1``)
and ``n_cuts_steps`` (all CUTS-active generated tokens). The reported percentage is:

    100 * sum(n_singleton) / sum(n_cuts_steps)

Usage:

    python analysis/stats.py --stats-dir /path/to/cuts_stats
    python analysis/stats.py --stats-dir /path/to/cuts_stats --step 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_records(stats_dir: Path, step: int | None = None) -> list[dict[str, Any]]:
    """Read complete JSON records from CUTS stats files."""
    records = []
    for path in sorted(stats_dir.glob("cuts_stats_*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A generation process can leave a partial trailing line.
                if line_number == len(lines):
                    continue
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from None
            if step is None or record.get("step") == step:
                records.append(record)
    return records


def singleton_token_stats(
    records: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    """Return singleton-token counts and percentage for CUTS records."""
    singleton_tokens = sum(int(record.get("n_singleton") or 0) for record in records)
    total_tokens = sum(int(record.get("n_cuts_steps") or 0) for record in records)
    return {
        "singleton_tokens": singleton_tokens,
        "total_cuts_tokens": total_tokens,
        "percentage": (
            100.0 * singleton_tokens / total_tokens if total_tokens else None
        ),
    }


def stats_by_step(records: list[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    """Return singleton-token statistics grouped by training step."""
    records_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "step" in record:
            records_by_step[int(record["step"])].append(record)
    return {
        str(step): singleton_token_stats(step_records)
        for step, step_records in sorted(records_by_step.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the percentage of CUTS tokens generated from singleton sets."
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="Directory containing cuts_stats_*.jsonl files.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Only include records for this training step.",
    )
    arguments = parser.parse_args()

    if not arguments.stats_dir.is_dir():
        parser.error(f"stats directory does not exist: {arguments.stats_dir}")

    records = _read_records(arguments.stats_dir, arguments.step)
    result = {
        "overall": singleton_token_stats(records),
        "by_step": stats_by_step(records) if arguments.step is None else {},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()