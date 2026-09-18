from __future__ import annotations

from mixed_cuts.boxed import last_answer_line, last_boxed_content, normalize_letter


def test_last_boxed_is_brace_balanced_and_takes_the_last_one():
    s = "First \\boxed{1} then \\boxed{\\frac{a}{b} + \\sqrt{2}} end."
    assert last_boxed_content(s) == "\\frac{a}{b} + \\sqrt{2}"
    assert last_boxed_content("no box") is None
    assert last_boxed_content("") is None
    assert last_boxed_content("truncated \\boxed{\\frac{1}{2") is None
    assert last_boxed_content("\\boxed{}") == ""


def test_answer_line_fallback():
    assert last_answer_line("... so\nAnswer: 42\n") == "42"
    assert last_answer_line("**Answer**: x = 3") == "x = 3"
    assert last_answer_line("Answer: 1\nAnswer: 2") == "2"
    assert last_answer_line("nothing") is None


def test_normalize_letter():
    assert normalize_letter("(B)") == "B"
    assert normalize_letter("c") == "C"
    assert normalize_letter("\\text{D}") == "D"
    assert normalize_letter("AB") is None
    assert normalize_letter(None) is None
