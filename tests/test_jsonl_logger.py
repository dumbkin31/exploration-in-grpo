from __future__ import annotations

import numpy as np
import torch

from mixed_cuts.jsonl_logger import JsonlMetricsLogger, to_jsonable


def test_log_and_read_round_trip(tmp_path):
    lg = JsonlMetricsLogger(tmp_path / "sub" / "metrics.jsonl")
    lg.log(
        {"a": np.float32(1.5), "b": torch.tensor(2), "c": np.arange(2), "d": {"e": torch.tensor([1.0])}},
        step=3,
    )
    lg.log({"a": 2.0}, step=4)
    recs = lg.read()
    assert [r["step"] for r in recs] == [3, 4]
    assert recs[0]["a"] == 1.5 and recs[0]["b"] == 2 and recs[0]["c"] == [0, 1] and recs[0]["d"]["e"] == [1.0]
    assert "time" in recs[0]


def test_to_jsonable_falls_back_to_str():
    class Weird:
        def __repr__(self):
            return "weird"

    assert to_jsonable(Weird()) == "weird"
    assert to_jsonable((1, "x", None)) == [1, "x", None]
