"""Unit tests for the pure CUTS operator (SELECT / FILTER / EQUALIZE)."""

from __future__ import annotations

import pytest
import torch

from cuts.operator import cuts_transform
from tests.conftest import logits_from_probs


def _uniform_over(out_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (survivor mask, softmax) of one output row."""
    probs = torch.softmax(out_row.float(), dim=-1)
    survivors = torch.isfinite(out_row)
    return survivors, probs


def test_k_larger_than_vocab_is_clamped():
    logits = logits_from_probs([0.5, 0.3, 0.15, 0.05])
    res = cuts_transform(logits, k=10, delta=0.1)
    assert int(res.set_size[0]) == 3  # 0.05 < 0.1 filtered; k clamped to vocab=4
    survivors, probs = _uniform_over(res.logits[0])
    assert survivors.tolist() == [True, True, True, False]
    assert torch.allclose(probs[survivors], torch.full((3,), 1 / 3))
    assert not bool(res.used_fallback[0])


def test_k_larger_than_surviving_set():
    # top-5 selected, only two candidates pass delta -> uniform over exactly those two
    logits = logits_from_probs([0.6, 0.3, 0.05, 0.03, 0.02])
    res = cuts_transform(logits, k=5, delta=0.1)
    assert int(res.set_size[0]) == 2
    survivors, probs = _uniform_over(res.logits[0])
    assert survivors.tolist() == [True, True, False, False, False]
    assert torch.allclose(probs[:2], torch.tensor([0.5, 0.5]))
    assert torch.all(probs[2:] == 0)


def test_everything_filtered_falls_back_to_topk_uniform():
    # delta above every probability: the paper's rule keeps the whole top-K set
    logits = logits_from_probs([0.4, 0.3, 0.2, 0.06, 0.04])
    res = cuts_transform(logits, k=3, delta=0.99, empty_set_fallback="topk")
    assert int(res.set_size[0]) == 3
    assert bool(res.used_fallback[0])
    survivors, probs = _uniform_over(res.logits[0])
    assert survivors.tolist() == [True, True, True, False, False]
    assert torch.allclose(probs[:3], torch.full((3,), 1 / 3))


def test_everything_filtered_falls_back_to_argmax():
    logits = logits_from_probs([0.4, 0.3, 0.2, 0.06, 0.04])
    res = cuts_transform(logits, k=3, delta=0.99, empty_set_fallback="argmax")
    assert int(res.set_size[0]) == 1
    assert bool(res.used_fallback[0])
    survivors, probs = _uniform_over(res.logits[0])
    assert survivors.tolist() == [True, False, False, False, False]
    assert probs[0] == pytest.approx(1.0)


@pytest.mark.parametrize("fallback", ["topk", "argmax"])
def test_set_is_never_empty_and_argmax_always_survives(fallback):
    logits = torch.randn(64, 300) * 3
    for delta in (0.0, 0.03, 0.5, 1.0):
        res = cuts_transform(logits, k=5, delta=delta, empty_set_fallback=fallback)
        assert torch.all(res.set_size >= 1)
        argmax = logits.argmax(dim=-1)
        assert torch.all(torch.isfinite(res.logits[torch.arange(64), argmax]))


def test_output_distribution_is_uniform_over_survivors():
    logits = torch.randn(16, 1000) * 2
    k, delta = 7, 0.02
    res = cuts_transform(logits, k=k, delta=delta)
    probs_in = torch.softmax(logits, dim=-1)
    topk_idx = torch.topk(probs_in, k, dim=-1).indices
    for b in range(16):
        survivors, probs_out = _uniform_over(res.logits[b])
        n = int(survivors.sum())
        assert n == int(res.set_size[b]) and 1 <= n <= k
        # uniform over survivors, exactly zero elsewhere
        assert torch.allclose(probs_out[survivors], torch.full((n,), 1 / n), atol=1e-6)
        assert torch.all(probs_out[~survivors] == 0)
        # survivors are top-k tokens that passed the threshold (unless the fallback fired)
        surv_idx = set(survivors.nonzero().squeeze(1).tolist())
        assert surv_idx <= set(topk_idx[b].tolist())
        if not bool(res.used_fallback[b]):
            assert all(probs_in[b, i] >= delta for i in surv_idx)
            rejected = set(topk_idx[b].tolist()) - surv_idx
            assert all(probs_in[b, i] < delta for i in rejected)


def test_inactive_rows_are_untouched_and_input_not_modified():
    logits = torch.randn(4, 50)
    original = logits.clone()
    mask = torch.tensor([True, False, True, False])
    res = cuts_transform(logits, k=3, delta=0.01, active_mask=mask)
    assert torch.equal(logits, original), "input must not be modified when inplace=False"
    assert torch.equal(res.logits[1], original[1]) and torch.equal(res.logits[3], original[3])
    assert res.set_size[1] == 0 and res.set_size[3] == 0
    assert res.set_size[0] >= 1 and res.set_size[2] >= 1
    assert torch.isinf(res.logits[0]).sum() >= 47  # at most 3 survivors
    # all-inactive batch is a pure pass-through
    res2 = cuts_transform(logits, k=3, delta=0.01, active_mask=torch.zeros(4, dtype=torch.bool))
    assert torch.equal(res2.logits, original) and torch.all(res2.set_size == 0)


def test_inplace_writes_into_the_input():
    logits = torch.randn(3, 20)
    res = cuts_transform(logits, k=2, delta=0.0, inplace=True)
    assert res.logits is logits
    assert torch.isinf(logits).sum() == 3 * 18


def test_fp16_dtype_is_preserved_and_small_probs_survive():
    # fp16 logits are what the 2080 Ti produces; the threshold must be evaluated in fp32.
    logits = logits_from_probs([0.9, 0.05, 0.03, 0.02], dtype=torch.float16)
    res = cuts_transform(logits, k=4, delta=0.025)
    assert res.logits.dtype == torch.float16
    assert int(res.set_size[0]) == 3
    survivors, probs = _uniform_over(res.logits[0])
    assert survivors.tolist() == [True, True, True, False]
    assert torch.allclose(probs[:3], torch.full((3,), 1 / 3), atol=1e-3)


def test_threshold_is_inclusive():
    logits = logits_from_probs([0.5, 0.25, 0.25])
    res = cuts_transform(logits, k=3, delta=0.25)
    assert int(res.set_size[0]) == 3


def test_invalid_arguments():
    logits = torch.randn(2, 10)
    with pytest.raises(ValueError):
        cuts_transform(logits, k=0, delta=0.1)
    with pytest.raises(ValueError):
        cuts_transform(logits, k=3, delta=0.1, empty_set_fallback="nope")
    with pytest.raises(ValueError):
        cuts_transform(logits.flatten(), k=3, delta=0.1)
    with pytest.raises(ValueError):
        cuts_transform(logits, k=3, delta=0.1, active_mask=torch.ones(3, dtype=torch.bool))
