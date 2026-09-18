"""Reward function tests (math-verify is in the dev env; the subprocess pool is exercised)."""

from __future__ import annotations

import pytest

from mc_data.schema import DS_GPQA, DS_MATH500, DS_MATH_TRAIN
from mixed_cuts import reward


@pytest.fixture(scope="module", autouse=True)
def _pool():
    yield
    reward.shutdown_pool()


@pytest.mark.parametrize(
    "gt,response,expected",
    [
        ("42", "blah blah \\boxed{42}", 1.0),
        ("42", "\\boxed{41}", 0.0),
        ("\\frac{1}{2}", "so the answer is \\boxed{0.5}", 1.0),
        ("\\frac{1}{2}", "so the answer is \\boxed{\\dfrac{1}{2}}", 1.0),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "\\boxed{\\left(3,\\frac{\\pi}{2}\\right)}", 1.0),
        ("x^2+1", "\\boxed{1 + x^2}", 1.0),
        ("10", "Answer: 10", 1.0),  # DAPO-style fallback without a box
        ("10", "no answer here", 0.0),
        ("3", "\\boxed{2} ... \\boxed{3}", 1.0),  # last box wins
    ],
)
def test_math_scores(gt, response, expected):
    out = reward.compute_score(DS_MATH_TRAIN, response, gt)
    assert out["score"] == expected
    assert set(out) == {"score", "pred", "has_boxed"}


def test_has_boxed_and_require_boxed():
    assert reward.compute_score(DS_MATH500, "Answer: 7", "7")["has_boxed"] == 0.0
    assert reward.compute_score(DS_MATH500, "Answer: 7", "7", require_boxed=True)["score"] == 0.0
    assert reward.compute_score(DS_MATH500, "\\boxed{7}", "7", require_boxed=True)["score"] == 1.0


def test_gpqa_letters():
    assert reward.compute_score(DS_GPQA, "The answer is \\boxed{B}", "B")["score"] == 1.0
    assert reward.compute_score(DS_GPQA, "\\boxed{(b)}", "B")["score"] == 1.0
    assert reward.compute_score(DS_GPQA, "\\boxed{A}", "B")["score"] == 0.0
    assert reward.compute_score(DS_GPQA, "Answer: C", "C")["score"] == 1.0
    assert reward.compute_score(DS_GPQA, "\\boxed{AB}", "A")["score"] == 0.0


def test_garbage_never_raises():
    out = reward.compute_score(DS_MATH500, "\\boxed{\\frac{}{}}\\", "\\frac{1}{2}")
    assert out["score"] in (0.0, 1.0)
    out = reward.compute_score(DS_MATH500, "\\boxed{" + "(" * 500 + "}", "1", timeout=5)
    assert out["score"] == 0.0


def test_verl_keyword_call_style():
    out = reward.compute_score(
        data_source=DS_MATH500, solution_str="\\boxed{9}", ground_truth="9", extra_info={"x": 1}
    )
    assert out["score"] == 1.0
