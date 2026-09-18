"""Unit tests for the per-request state table (prefix protection, isolation, cleanup).

The ``BatchUpdate`` objects here are hand-built with the exact tuple shapes vLLM 0.24.0 uses
(``vllm/v1/sample/logits_processor/interface.py``): ``added`` entries are
``(index, params, prompt_tok_ids, output_tok_ids)`` and ``moved`` entries are
``(from_index, to_index, direction)``. ``output_tok_ids`` is a live list, as in vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from cuts.config import CutsParams
from cuts.state import CutsBatchState


class MoveDirectionality(Enum):  # local stand-in for vLLM's enum (compared by name)
    UNIDIRECTIONAL = auto()
    SWAP = auto()


@dataclass
class FakeSamplingParams:
    extra_args: dict[str, Any] | None = None


@dataclass
class BatchUpdate:
    batch_size: int
    removed: list[int] = field(default_factory=list)
    added: list[tuple] = field(default_factory=list)
    moved: list[tuple] = field(default_factory=list)


def cuts_request(**kw) -> FakeSamplingParams:
    return FakeSamplingParams(extra_args=CutsParams(**kw).to_extra_args())


def std_request() -> FakeSamplingParams:
    return FakeSamplingParams(extra_args=None)


def is_uniform_cuts_row(row: torch.Tensor, max_size: int) -> bool:
    finite = torch.isfinite(row)
    n = int(finite.sum())
    return 1 <= n <= max_size and torch.all(row[finite] == 0).item()


def test_noop_before_t_warm_and_active_after():
    state = CutsBatchState()
    out_ids: list[int] = []
    state.apply_batch_update(
        BatchUpdate(1, added=[(0, cuts_request(k=3, delta=0.0, t_warm=3), [1, 2], out_ids)])
    )
    logits = torch.randn(1, 30)
    for step in range(3):  # 0, 1, 2 generated tokens: standard sampling
        assert len(out_ids) == step
        assert torch.equal(state.apply(logits.clone()), logits), f"CUTS must be a no-op at step {step}"
        assert state.active_indices() == []
        out_ids.append(100 + step)  # vLLM appends the sampled token to the live list
    assert state.active_indices() == [0]
    out = state.apply(logits.clone())
    assert is_uniform_cuts_row(out[0], max_size=3)
    entry = state.get(0)
    assert entry is not None and entry.n_cuts_steps == 1 and entry.n_generated == 3


def test_t_warm_zero_is_active_immediately():
    state = CutsBatchState()
    state.apply_batch_update(BatchUpdate(1, added=[(0, cuts_request(k=2, delta=0.0, t_warm=0), [], [])]))
    out = state.apply(torch.randn(1, 10))
    assert is_uniform_cuts_row(out[0], max_size=2)


def test_standard_requests_are_never_tracked_or_touched():
    state = CutsBatchState()
    state.apply_batch_update(BatchUpdate(2, added=[(0, std_request(), [], []), (1, std_request(), [], [])]))
    assert len(state) == 0
    logits = torch.randn(2, 20)
    assert state.apply(logits.clone()) is not None
    assert torch.equal(state.apply(logits.clone()), logits)


def test_interleaved_requests_keep_isolated_state():
    finished: list[dict] = []
    state = CutsBatchState(on_request_finished=finished.append)
    ids0, ids2 = [1, 2, 3], [7, 8, 9]  # already past warm-up (t_warm=2)
    state.apply_batch_update(
        BatchUpdate(
            3,
            added=[
                (0, cuts_request(k=2, delta=0.0, t_warm=2, uid="p0", session_id=5), [], ids0),
                (1, std_request(), [], []),
                (2, cuts_request(k=5, delta=0.0, t_warm=2, uid="p2", session_id=9), [], ids2),
            ],
        )
    )
    logits = torch.randn(3, 40)
    out = state.apply(logits.clone())
    assert is_uniform_cuts_row(out[0], max_size=2) and int(torch.isfinite(out[0]).sum()) == 2
    assert torch.equal(out[1], logits[1]), "the standard request in the middle must be untouched"
    assert is_uniform_cuts_row(out[2], max_size=5) and int(torch.isfinite(out[2]).sum()) == 5

    # request 0 finishes; a NEW standard request reuses row 0 in the same update
    state.apply_batch_update(BatchUpdate(3, removed=[0], added=[(0, std_request(), [], [])]))
    assert len(finished) == 1
    assert finished[0]["uid"] == "p0" and finished[0]["session_id"] == 5
    assert finished[0]["n_cuts_steps"] == 1 and finished[0]["mean_set_size"] == 2.0
    assert 0 not in state and 2 in state and len(state) == 1
    out = state.apply(logits.clone())
    assert torch.equal(out[0], logits[0]), "row 0 now belongs to a standard request"
    assert is_uniform_cuts_row(out[2], max_size=5)

    # vLLM condenses: request at row 2 moves into the vacated row 1 (UNIDIRECTIONAL)
    state.apply_batch_update(BatchUpdate(2, moved=[(2, 1, MoveDirectionality.UNIDIRECTIONAL)]))
    assert 1 in state and 2 not in state and state.get(1).params.uid == "p2"
    out = state.apply(logits[:2].clone())
    assert torch.equal(out[0], logits[0]) and is_uniform_cuts_row(out[1], max_size=5)
    assert state.get(1).n_cuts_steps == 3


def test_swap_exchanges_rows():
    state = CutsBatchState()
    state.apply_batch_update(
        BatchUpdate(
            2,
            added=[
                (0, cuts_request(k=1, delta=0.0, t_warm=0, uid="a"), [], []),
                (1, cuts_request(k=4, delta=0.0, t_warm=0, uid="b"), [], []),
            ],
        )
    )
    state.apply_batch_update(BatchUpdate(2, moved=[(0, 1, MoveDirectionality.SWAP)]))
    assert state.get(0).params.uid == "b" and state.get(1).params.uid == "a"
    out = state.apply(torch.randn(2, 10))
    assert int(torch.isfinite(out[0]).sum()) == 4 and int(torch.isfinite(out[1]).sum()) == 1
    # swap with an empty row just moves the occupant
    state.apply_batch_update(BatchUpdate(3, moved=[(1, 2, MoveDirectionality.SWAP)]))
    assert 1 not in state and state.get(2).params.uid == "a"


def test_removal_flushes_stats_and_leaves_no_state_behind():
    finished: list[dict] = []
    state = CutsBatchState(on_request_finished=finished.append)
    ids = [1, 2, 3, 4, 5]
    state.apply_batch_update(
        BatchUpdate(1, added=[(0, cuts_request(k=3, delta=0.99, t_warm=0, step=7), [1], ids)])
    )
    logits = torch.randn(1, 10)
    for _ in range(4):
        state.apply(logits.clone())
    state.apply_batch_update(BatchUpdate(0, removed=[0]))
    assert len(state) == 0 and state.active_indices() == []
    assert len(finished) == 1
    rec = finished[0]
    assert rec["step"] == 7 and rec["prompt_len"] == 1 and rec["n_generated"] == 5
    assert rec["n_cuts_steps"] == 4 and rec["n_fallback"] == 4  # delta=0.99 always triggers the fallback
    assert rec["mean_set_size"] == 3.0 and rec["n_singleton"] == 0
    # the row can be reused by a fresh CUTS request with fresh counters
    state.apply_batch_update(BatchUpdate(1, added=[(0, cuts_request(k=3, delta=0.0, t_warm=0), [], [])]))
    assert state.get(0).n_cuts_steps == 0
    # removing an untracked row (a standard request) is a no-op
    state.apply_batch_update(BatchUpdate(1, removed=[5]))
    assert len(finished) == 1


def test_added_request_replaces_previous_occupant_and_flushes_it():
    finished: list[dict] = []
    state = CutsBatchState(on_request_finished=finished.append)
    state.apply_batch_update(BatchUpdate(1, added=[(0, cuts_request(uid="old", t_warm=0), [], [])]))
    state.apply_batch_update(BatchUpdate(1, added=[(0, cuts_request(uid="new", t_warm=0), [], [])]))
    assert [r["uid"] for r in finished] == ["old"] and state.get(0).params.uid == "new"


def test_singleton_and_argmax_fallback_stats():
    state = CutsBatchState()
    state.apply_batch_update(
        BatchUpdate(
            1, added=[(0, cuts_request(k=5, delta=0.99, t_warm=0, empty_set_fallback="argmax"), [], [])]
        )
    )
    for _ in range(3):
        state.apply(torch.randn(1, 10))
    e = state.get(0)
    assert e.n_cuts_steps == 3 and e.n_singleton == 3 and e.n_fallback == 3 and e.sum_set_size == 3


def test_clear_finishes_everything():
    finished: list[dict] = []
    state = CutsBatchState(on_request_finished=finished.append)
    state.apply_batch_update(BatchUpdate(2, added=[(0, cuts_request(), [], []), (1, cuts_request(), [], [])]))
    state.clear()
    assert len(state) == 0 and len(finished) == 2


def test_none_update_is_ignored():
    state = CutsBatchState()
    state.apply_batch_update(None)
    assert len(state) == 0
