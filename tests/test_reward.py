"""Reward function tests (math-verify is in the dev env; the subprocess pool is exercised)."""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from mc_data.schema import DS_GPQA, DS_MATH500, DS_MATH_TRAIN
from mixed_cuts import reward

HALF = (DS_MATH500, "so the answer is \\boxed{0.5}", "\\frac{1}{2}")  # equal only under math-verify


@pytest.fixture(scope="module", autouse=True)
def _pool():
    yield
    reward.shutdown_pool()


@pytest.fixture
def fresh_pool(monkeypatch):
    """A pool built inside the test, so env vars set with the returned monkeypatch reach its workers."""
    reward.shutdown_pool()
    yield monkeypatch
    reward.shutdown_pool()


# Stand-ins for reward._math_verify_equal. They run in the spawned workers, which import them from
# this module by name, so they must live at module level.
def _exit_on_first_call(gold: str, pred: str) -> bool:
    marker = Path(os.environ["MC_TEST_MARKER"])
    if not marker.exists():
        marker.touch()
        os._exit(1)  # the worker dies mid-comparison, like an OOM kill or a segfault
    return reward._math_verify_equal(gold, pred)


def _always_exit(gold: str, pred: str) -> bool:
    os._exit(1)


def _slow_equal(gold: str, pred: str) -> bool:
    time.sleep(1.0)
    return reward._math_verify_equal(gold, pred)


def _hang(gold: str, pred: str) -> bool:
    Path(os.environ["MC_TEST_MARKER"]).write_text(str(os.getpid()))
    time.sleep(600)  # stuck where math-verify's alarm cannot reach
    return True


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


def test_broken_pool_is_replaced(fresh_pool):
    """A dead worker breaks a ProcessPoolExecutor for good; grading must not fall back to exact match."""
    with pytest.raises(BrokenProcessPool):
        reward._get_pool().submit(os._exit, 1).result(timeout=60)
    before = reward.pool_restarts()
    for _ in range(2):
        assert reward.compute_score(*HALF)["score"] == 1.0
    assert reward.pool_restarts() == before + 1


def test_worker_death_mid_comparison_is_retried(fresh_pool, tmp_path):
    fresh_pool.setenv("MC_TEST_MARKER", str(tmp_path / "died"))
    fresh_pool.setattr(reward, "_math_verify_equal", _exit_on_first_call)
    before = reward.pool_restarts()
    assert reward.compute_score(*HALF)["score"] == 1.0
    assert reward.pool_restarts() == before + 1


def test_answer_that_kills_every_worker_scores_zero_with_a_warning(fresh_pool, caplog):
    fresh_pool.setattr(reward, "_math_verify_equal", _always_exit)
    before = reward.pool_restarts()
    with caplog.at_level(logging.WARNING, logger=reward.__name__):
        assert reward.compute_score(*HALF)["score"] == 0.0
    assert reward.pool_restarts() == before + 2  # one retry, then give up
    assert "died on the retry" in caplog.text


def test_queue_wait_does_not_count_against_timeout(fresh_pool):
    """verl runs up to 32 compute_score calls per process at once (module docstring)."""
    fresh_pool.setenv("MC_REWARD_WORKERS", "1")
    fresh_pool.setattr(reward, "_math_verify_equal", _slow_equal)  # 1 s per comparison
    assert reward.compute_score(*HALF)["score"] == 1.0  # warm-up: spawn + imports
    with ThreadPoolExecutor(4) as callers:
        scores = list(callers.map(lambda _: reward.compute_score(*HALF, timeout=2.5)["score"], range(4)))
    # the 3rd and 4th calls wait 2 s and 3 s for the worker; only their 1 s of running counts
    assert scores == [1.0] * 4


def test_stuck_worker_is_killed_and_replaced(fresh_pool, tmp_path):
    pid_file = tmp_path / "pid"
    real_equal = reward._math_verify_equal
    fresh_pool.setenv("MC_REWARD_WORKERS", "1")
    fresh_pool.setenv("MC_TEST_MARKER", str(pid_file))
    fresh_pool.setattr(reward, "_math_verify_equal", _hang)
    before = reward.pool_restarts()
    assert reward.compute_score(*HALF, timeout=5)["score"] == 0.0
    assert reward.pool_restarts() == before + 1

    # the only worker is stuck: grading works again only because the pool was replaced
    fresh_pool.setattr(reward, "_math_verify_equal", real_equal)
    assert reward.compute_score(*HALF)["score"] == 1.0

    stuck_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 30
    while stuck_pid in {p.pid for p in multiprocessing.active_children()} and time.monotonic() < deadline:
        time.sleep(0.1)
    assert stuck_pid not in {p.pid for p in multiprocessing.active_children()}


def test_zero_workers_fails_loudly(fresh_pool):
    fresh_pool.setenv("MC_REWARD_WORKERS", "0")
    with pytest.raises(ValueError, match="MC_REWARD_WORKERS"):
        reward.compute_score(*HALF)
