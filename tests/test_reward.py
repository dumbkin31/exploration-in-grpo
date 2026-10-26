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
        ("10", "Answer: 10", 0.0),  # no box -> invalid (Appendix B.4); no Answer:-line fallback for math
        ("10", "no answer here", 0.0),
        ("3", "\\boxed{2} ... \\boxed{3}", 1.0),  # last box wins
    ],
)
def test_math_scores(gt, response, expected):
    out = reward.compute_score(DS_MATH_TRAIN, response, gt)
    assert out["score"] == expected
    assert set(out) == {"score", "pred", "has_boxed", "valid"}


@pytest.mark.parametrize(
    "gt,response,expected",
    [
        # integers
        ("42", "\\boxed{ 42 }", 1.0),
        ("-7", "\\boxed{-7}", 1.0),
        ("1000", "\\boxed{1,000}", 1.0),
        # fractions
        ("\\frac{3}{4}", "\\boxed{\\frac{3}{4}}", 1.0),
        ("\\frac{3}{4}", "\\boxed{\\dfrac{3}{4}}", 1.0),
        ("\\frac{3}{4}", "\\boxed{0.75}", 1.0),
        ("\\frac{3}{4}", "\\boxed{3/4}", 1.0),
        ("\\frac{3}{4}", "\\boxed{\\frac{6}{8}}", 1.0),
        ("\\frac34", "\\boxed{\\frac{3}{4}}", 1.0),
        # square roots
        ("2\\sqrt{5}", "\\boxed{2\\sqrt{5}}", 1.0),
        ("2\\sqrt{5}", "\\boxed{\\sqrt{20}}", 1.0),
        ("2\\sqrt{5}", "\\boxed{2\\sqrt5}", 1.0),
        ("\\frac{\\sqrt{3}}{2}", "\\boxed{\\frac{1}{2}\\sqrt{3}}", 1.0),
        # symbolic answers without a digit: graded since decision 017 (invalid under the old digit rule)
        ("\\pi", "\\boxed{\\pi}", 1.0),
        ("\\infty", "\\boxed{\\infty}", 1.0),
        ("x^2+1", "\\boxed{1+x^2}", 1.0),
        ("\\text{(C)}", "\\boxed{C}", 1.0),
        ("\\text{odd}", "\\boxed{\\text{odd}}", 1.0),
        # wrong answers
        ("42", "\\boxed{41}", 0.0),
        ("\\frac{3}{4}", "\\boxed{\\frac{4}{3}}", 0.0),
        ("2\\sqrt{5}", "\\boxed{2\\sqrt{3}}", 0.0),
        ("\\pi", "\\boxed{2\\pi}", 0.0),
        ("-\\infty", "\\boxed{\\infty}", 0.0),
        ("\\text{odd}", "\\boxed{\\text{even}}", 0.0),
    ],
)
def test_symbolic_grading(gt, response, expected):
    out = reward.compute_score(DS_MATH_TRAIN, f"Some reasoning. So the answer is {response}.", gt)
    assert out["score"] == expected
    assert out["valid"] == 1.0 and out["has_boxed"] == 1.0


def test_extract_answer_keeps_any_nonempty_final_box():
    """Decision 017: valid = a final \\boxed{} with non-empty content; a digit is no longer required."""
    assert reward.extract_answer("so \\boxed{\\frac{3}{4}}") == ("\\frac{3}{4}", True, True)
    assert reward.extract_answer("\\boxed{2\\sqrt{5}}") == ("2\\sqrt{5}", True, True)
    assert reward.extract_answer("\\boxed{\\pi}") == ("\\pi", True, True)
    assert reward.extract_answer("\\boxed{\\frac{1}{\\sqrt{2}}}") == ("\\frac{1}{\\sqrt{2}}", True, True)
    # the last box wins and its content is stripped
    assert reward.extract_answer("\\boxed{1} then \\boxed{ x+1 }") == ("x+1", True, True)
    assert reward.extract_answer("no box, Answer: 7") == (None, False, False)
    assert reward.extract_answer("truncated \\boxed{\\frac{1}{2") == (None, False, False)
    assert reward.extract_answer("\\boxed{}") == ("", True, False)
    assert reward.extract_answer("\\boxed{   }") == ("", True, False)


def test_empty_and_missing_answers_are_invalid_and_score_zero():
    for response in ("\\boxed{}", "\\boxed{  }"):
        assert reward.compute_score(DS_MATH500, response, "0") == {
            "score": 0.0,
            "pred": "",
            "has_boxed": 1.0,
            "valid": 0.0,
        }
    assert reward.compute_score(DS_MATH500, "Answer: 7", "7") == {
        "score": 0.0,
        "pred": "",
        "has_boxed": 0.0,
        "valid": 0.0,
    }


@pytest.mark.parametrize(
    "pred", ["\\frac{}{}", "\\undefinedmacro{x}", "??", "\\left(", "\\begin{matrix}", "}{", "\\sqrt{"]
)
def test_unparseable_answers_score_zero_without_raising(pred):
    out = reward.compute_score(DS_MATH500, f"\\boxed{{{pred}}}", "\\pi", timeout=10)
    assert out["score"] == 0.0


def test_has_digit_is_only_the_training_filter():
    """prepare_data.py still drops digit-less ground truths (decision 008); the reward no longer uses it."""
    assert reward.has_digit("\\frac{1}{2}") and not reward.has_digit("e") and not reward.has_digit(None)


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
