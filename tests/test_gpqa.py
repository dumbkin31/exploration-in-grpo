"""Five realistic Qwen-style GPQA outputs + edge cases for the letter parser (Task C)."""

from __future__ import annotations

import pytest

from mc_data.schema import DS_GPQA
from mixed_cuts import reward
from mixed_cuts.gpqa import extract_choice_letter

QWEN_OUTPUTS = [
    # 1. boxed letter, the requested format
    (
        "Let's analyse each option.\n\n(A) ... is wrong because ...\n(B) ... \n\nThe correct option is (C).\n\n\\boxed{C}",
        "C",
    ),
    # 2. boxed with parentheses and \text
    ("Therefore the answer is \\boxed{\\text{(B)}}.", "B"),
    # 3. no box at all, explicit statement with markdown
    ("...so the reaction proceeds via an SN2 pathway.\n\n**Answer:** (D)", "D"),
    # 4. boxed contains the letter and the option text
    ("Thus, \\boxed{A) 3.2 \\times 10^{-5} \\text{ M}}", "A"),
    # 5. no box, no 'answer' keyword: final standalone letter, article 'A' earlier must not win
    (
        "A quick check of the units shows option B fails. Comparing C and D, only the second holds.\nFinal choice: D.",
        "D",
    ),
]


@pytest.mark.parametrize("text,expected", QWEN_OUTPUTS)
def test_realistic_qwen_outputs(text, expected):
    assert extract_choice_letter(text) == expected


def test_fallback_order_and_edge_cases():
    assert extract_choice_letter("the answer is (C) but \\boxed{B}") == "B"  # final box wins
    assert extract_choice_letter("Option A is a trap; the answer is C") == "C"  # explicit beats standalone
    assert extract_choice_letter("A cat sat.") is None  # article only
    assert extract_choice_letter("") is None and extract_choice_letter("no letters here 42") is None
    assert extract_choice_letter("\\boxed{42}") is None  # a number is not a choice
    assert extract_choice_letter("\\boxed{AB}") is None
    assert extract_choice_letter("choice: b") == "B"


def test_gpqa_reward_uses_the_letter_parser():
    assert reward.compute_score(DS_GPQA, "The answer is \\boxed{B}", "B") == {
        "score": 1.0,
        "pred": "B",
        "has_boxed": 1.0,
        "valid": 1.0,
    }
    assert reward.compute_score(DS_GPQA, "**Answer:** (C)", "c")["score"] == 1.0
    assert reward.compute_score(DS_GPQA, "nothing", "C") == {
        "score": 0.0,
        "pred": "",
        "has_boxed": 0.0,
        "valid": 0.0,
    }
