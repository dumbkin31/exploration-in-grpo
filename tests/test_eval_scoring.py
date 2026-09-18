from __future__ import annotations

import pytest

from mc_eval.scoring import (
    aggregate,
    cluster_answers,
    majority_answer,
    normalize_answer_string,
    pass_at_k,
    summarize_problem,
)


def test_pass_at_k_estimator():
    assert pass_at_k(16, 0, 1) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert pass_at_k(16, 4, 1) == pytest.approx(0.25)
    assert pass_at_k(16, 1, 16) == 1.0
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)  # 1 - C(3,2)/C(4,2) = 1 - 3/6
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


def test_normalize_and_cluster():
    assert normalize_answer_string(" \\dfrac{1}{2} ") == "\\frac{1}{2}"
    assert normalize_answer_string("\\left( 3, \\frac{\\pi}{2} \\right)") == "(3,\\frac{\\pi}{2})"
    assert normalize_answer_string("90^\\circ") == "90"
    assert normalize_answer_string(None) == ""
    answers = ["\\frac{1}{2}", "0.5", "\\dfrac{1}{2}", None, "", "3"]
    clusters = cluster_answers(answers)  # no expensive equality: 1/2 and 0.5 stay apart
    assert clusters == [[0, 2], [1], [3], [4], [5]]
    clusters = cluster_answers(answers, equal=lambda a, b: {a, b} <= {"\\frac{1}{2}", "0.5"})
    assert clusters[0] == [0, 1, 2]


def test_majority_answer_and_ties():
    maj, size = majority_answer(["a", "b", "b", "c"])
    assert (maj, size) == ("b", 2)
    maj, size = majority_answer(["a", "b"])  # tie -> earliest
    assert (maj, size) == ("a", 1)
    assert majority_answer([None, None, "x"]) == ("x", 1) or majority_answer([None, None, "x"]) == (None, 1)


def test_summarize_problem_and_aggregate():
    correct = [1, 1, 0, 0]
    answers = ["7", "7", "8", None]
    s = summarize_problem(correct, answers, "7", k_values=(1, 4), is_correct_answer=lambda a: a == "7")
    assert s["pass@1"] == 0.5 and s["pass@4"] == 1.0 and s["maj@4"] == 1.0
    assert s["maj@4_agreement"] == 0.5 and s["frac_no_answer"] == 0.25 and s["maj@1"] == 1.0
    agg = aggregate([s, {**s, "pass@1": 0.0, "maj@4": 0.0}], n_boot=200)
    assert agg["n_problems"] == 2 and agg["pass@1"] == 0.25 and agg["maj@4"] == 0.5
    assert len(agg["pass@1_ci95"]) == 2 and agg["pass@1_ci95"][0] <= agg["pass@1"] <= agg["pass@1_ci95"][1]
    assert aggregate([]) == {"n_problems": 0}
