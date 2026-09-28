"""The memory sidecar splits each process's RSS into anonymous and shared (pinned) memory from /proc."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("profile_memory", REPO / "scripts" / "profile_memory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_breakdown_reads_rss_anon_and_shmem(tmp_path: Path):
    mod = _load()
    (tmp_path / "123").mkdir()
    (tmp_path / "123" / "status").write_text(
        "Name:\tpython\nVmRSS:\t 19000000 kB\nRssAnon:\t 7340032 kB\nRssFile:\t 1048576 kB\nRssShmem:\t 11534336 kB\n"
    )
    assert mod.proc_rss_breakdown("123", proc=tmp_path) == (7168, 11264)


def test_breakdown_missing_process_is_none(tmp_path: Path):
    mod = _load()
    assert mod.proc_rss_breakdown("999", proc=tmp_path) == (None, None)
