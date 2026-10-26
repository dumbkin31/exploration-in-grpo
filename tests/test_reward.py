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


def test_validity_rule_boxed_and_digit():
    """A response is valid only if the FINAL \\boxed{} exists and its content has a digit."""
    assert reward.compute_score(DS_MATH500, "Answer: 7", "7") == {
        "score": 0.0,
        "pred": "",
        "has_boxed": 0.0,
        "valid": 0.0,
    }
    out = reward.compute_score(DS_MATH500, "\\boxed{\\pi}", "\\pi")  # equivalent but digit-less -> invalid
    assert out == {"score": 0.0, "pred": "\\pi", "has_boxed": 1.0, "valid": 0.0}
    assert reward.compute_score(DS_MATH500, "\\boxed{}", "0")["valid"] == 0.0
    out = reward.compute_score(DS_MATH500, "\\boxed{2\\pi}", "2\\pi")
    assert out["valid"] == 1.0 and out["score"] == 1.0
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


# --- equivalence by value, not text: (gold, model answer); a bare answer is put in \boxed{} ---

ONE_HALF_FORMS = ["1/2", "0.5", "\\frac{1}{2}", "\\dfrac{1}{2}", ".5", "0.50", "\\dfrac12", "2/4"]

EQUAL_VALUE = [
    ("1/2", "0.5"),
    ("1/2", "\\frac{1}{2}"),
    ("1/2", "\\dfrac12"),
    ("1/2", ".5"),
    ("1/2", "0.50"),
    ("1/2", "The answer is \\boxed{0.5}"),
    ("1/2", "2/4"),
    ("0.75", "3/4"),
    ("0.75", "\\frac{3}{4}"),
    ("-3/2", "-1.5"),
    ("2", "2.0"),
    ("2", "4/2"),
    ("1000", "1,000"),
    ("1000", "1000.0"),
    ("2\\sqrt{5}", "\\sqrt{20}"),
    ("0.25", "1/4"),
    ("1/4", "0.25"),
    ("\\frac{1}{3}", "0.333333"),  # within reward.FLOAT_ROUNDING = 6 decimal places
    # a percent sign is a unit label (reward._strip_percent, decided 2026-10-26)
    ("10", "10\\%"),
    ("73\\%", "73"),
    ("12.5\\%", "12.50\\%"),
    ("50\\%", "50.0 \\%"),
]

DIFFERENT_VALUE = [
    ("1/2", "0.25"),
    ("1/2", "1/3"),
    ("1/2", "2"),
    ("0.5", "1/3"),
    ("0.5", "5"),
    ("5", "0.5"),
    ("0.75", "0.7"),
    ("-3/2", "1.5"),
    ("2", "2.5"),
    ("10\\%", "0.1\\%"),
    ("10", "11\\%"),
    # rounded non-terminating decimals: kept strict on 2026-10-26 (reward.FLOAT_ROUNDING)
    ("1/3", "0.333"),
    ("1/3", "0.33"),
    ("2\\sqrt{5}", "4.47"),
]

# Not settled: a percent read as a fraction (50% = 0.5). Only "binary score, no exception" is
# asserted here.
REPORT_ONLY = [
    ("50", "0.5"),
    ("10\\%", "0.1"),
]

GARBAGE = [
    "",
    "no box at all",
    "\\boxed{}",
    "\\boxed{asdf}",
    "\\boxed{\\frac{}{}}",
    "\\boxed{1/0}",
    "\\boxed{)(*&^%1}",
    "\\boxed{\\frac{1}{2",  # unterminated box
    "\\boxed{1\\%\\%}",
]


def _boxed(answer: str) -> str:
    return answer if "\\boxed" in answer else f"\\boxed{{{answer}}}"


@pytest.mark.parametrize("gold,answer", EQUAL_VALUE)
def test_equal_values_in_other_formats_score_1(gold, answer):
    assert reward.compute_score(DS_MATH500, _boxed(answer), gold)["score"] == 1.0


@pytest.mark.parametrize("gold,answer", DIFFERENT_VALUE)
def test_different_values_score_0(gold, answer):
    assert reward.compute_score(DS_MATH500, _boxed(answer), gold)["score"] == 0.0


@pytest.mark.parametrize("gold", ONE_HALF_FORMS)
def test_every_form_of_one_half_matches_every_other(gold):
    scores = {a: reward.compute_score(DS_MATH500, _boxed(a), gold)["score"] for a in ONE_HALF_FORMS}
    assert scores == dict.fromkeys(ONE_HALF_FORMS, 1.0)


@pytest.mark.parametrize("gold,answer", REPORT_ONLY)
def test_unsettled_policy_cases_are_binary(gold, answer):
    assert reward.compute_score(DS_MATH500, _boxed(answer), gold)["score"] in (0.0, 1.0)


@pytest.mark.parametrize("response", [*GARBAGE, None])
def test_unparseable_answers_score_0_without_raising(response):
    out = reward.compute_score(DS_MATH500, response, "1/2", timeout=10)
    assert out["score"] == 0.0 and set(out) == {"score", "pred", "has_boxed", "valid"}


def test_math_verify_cannot_parse_percent_under_antlr_4_9():
    """Why _strip_percent exists. If this starts failing, the environment changed: revisit it."""
    from importlib.metadata import version

    from math_verify.parser import LatexExtractionConfig, parse

    if not version("antlr4-python3-runtime").startswith("4.9"):
        pytest.skip("only the ANTLR 4.9.3 parser of latex2sympy2_extended 1.11.0 lacks FUNC_GAMMA")
    parsed = parse("\\boxed{50\\%}", (LatexExtractionConfig(),))
    assert parsed and all(isinstance(x, str) for x in parsed)
