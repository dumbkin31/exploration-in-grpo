from __future__ import annotations

from mc_data import dapo, eval_sets, math
from mc_data.schema import BOXED_INSTRUCTION, DS_AIME24, DS_AMC23, DS_GPQA, DS_MATH500, Row

DAPO_PRE = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_POST = '\n\nRemember to put your answer on its own line after "Answer:".'


def _dapo_row(idx: str, q: str, gt: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": DAPO_PRE + q + DAPO_POST}],
        "ability": "MATH",
        "reward_model": {"ground_truth": gt, "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": idx},
    }


def test_row_record_schema():
    rec = Row("src", " q ", "7", "train", 3).to_record()
    assert rec["prompt"] == [{"role": "user", "content": f"q\n\n{BOXED_INSTRUCTION}"}]
    assert rec["reward_model"] == {"style": "rule", "ground_truth": "7"}
    assert rec["extra_info"] == {"split": "train", "index": "3"}
    assert rec["data_source"] == "src" and rec["ability"] == "math"


def test_dapo_dedupe_keeps_first_occurrence_and_counts_total():
    rows = [_dapo_row("a", "Q1", "1")] * 100 + [_dapo_row("b", "Q2", "2")] * 100 + [_dapo_row("a", "Q1", "1")]
    unique, total = dapo.deduplicate(rows)
    assert total == 201 and [r["extra_info"]["index"] for r in unique] == ["a", "b"]
    recs = [r.to_record() for r in dapo.rows_from_hf(unique)]
    assert recs[0]["prompt"][0]["content"] == f"Q1\n\n{BOXED_INSTRUCTION}"
    assert recs[0]["extra_info"]["original_prompt"].startswith(DAPO_PRE)
    assert recs[1]["reward_model"]["ground_truth"] == "2" and recs[1]["extra_info"]["index"] == "b"


def test_dapo_dedupe_without_index_falls_back_to_prompt_hash():
    rows = [{"prompt": [{"role": "user", "content": "x"}]}] * 3 + [
        {"prompt": [{"role": "user", "content": "y"}]}
    ]
    unique, total = dapo.deduplicate(rows)
    assert total == 4 and len(unique) == 2


def test_math_rows_extract_last_boxed_and_skip_unboxed():
    ds = [
        {"problem": "P1", "solution": "... \\boxed{\\frac{1}{2}}.", "level": "Level 1", "type": "Algebra"},
        {"problem": "P2", "solution": "no box", "level": "Level 2", "type": "Algebra"},
    ]
    rows = list(math.rows_from_hf(ds, "train"))
    assert len(rows) == 1 and rows[0].ground_truth == "\\frac{1}{2}" and rows[0].index == "train-0"


def test_eval_set_conversions():
    m = list(
        eval_sets.rows_from_hf(
            DS_MATH500, [{"problem": "p", "answer": "1", "unique_id": "u", "subject": "s", "level": 2}]
        )
    )
    assert m[0].ground_truth == "1" and m[0].index == "u"
    a = list(
        eval_sets.rows_from_hf(DS_AIME24, [{"id": 5, "problem": "p", "solution": "\\boxed{204}", "url": ""}])
    )
    assert a[0].ground_truth == "204" and a[0].index == "5"
    c = list(eval_sets.rows_from_hf(DS_AMC23, [{"id": 0, "question": "q", "answer": "27", "url": ""}]))
    assert c[0].question == "q" and c[0].ground_truth == "27"


def test_gpqa_shuffle_is_deterministic_and_letter_is_correct():
    ex = {
        "Question": "Why?",
        "Correct Answer": "right",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
        "Record ID": "r1",
    }
    r1 = list(eval_sets.rows_from_hf(DS_GPQA, [ex]))[0]
    r2 = list(eval_sets.rows_from_hf(DS_GPQA, [ex]))[0]
    assert r1.question == r2.question and r1.ground_truth == r2.ground_truth
    letter = r1.ground_truth
    assert f"({letter}) right" in r1.question and r1.ability == "science"
