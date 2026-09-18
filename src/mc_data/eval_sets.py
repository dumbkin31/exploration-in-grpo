"""The five evaluation benchmarks, each converted to the same verl row schema.

Hub column layouts (verified via the datasets-server API, 2026-09):

- ``HuggingFaceH4/MATH-500``      split ``test``: problem, solution, answer, subject, level, unique_id
- ``math-ai/aime24``              split ``test``: id, problem, solution (``\\boxed{...}``), url
- ``math-ai/aime25``              split ``test``: problem, answer, id
- ``math-ai/amc23``               split ``test``: id, question, answer, url
- ``Idavidrein/gpqa``             config ``gpqa_diamond``, split ``train``: Question, Correct Answer,
                                  Incorrect Answer 1/2/3, ... (GATED: needs HF_TOKEN)

GPQA is multiple choice: options are shuffled with a fixed seed per question, the letter of the
correct option is the ground truth, and the prompt asks for the letter in ``\\boxed{}``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator

from mc_data.schema import DS_AIME24, DS_AIME25, DS_AMC23, DS_GPQA, DS_MATH500, Row
from mixed_cuts.boxed import last_boxed_content

HF_SPECS = {
    DS_MATH500: ("HuggingFaceH4/MATH-500", None, "test"),
    DS_AIME24: ("math-ai/aime24", None, "test"),
    DS_AIME25: ("math-ai/aime25", None, "test"),
    DS_AMC23: ("math-ai/amc23", None, "test"),
    DS_GPQA: ("Idavidrein/gpqa", "gpqa_diamond", "train"),
}

GPQA_LETTERS = "ABCD"


def _gpqa_question(ex: dict, rng: random.Random) -> tuple[str, str]:
    options = [
        ex["Correct Answer"],
        ex["Incorrect Answer 1"],
        ex["Incorrect Answer 2"],
        ex["Incorrect Answer 3"],
    ]
    options = [str(o).strip() for o in options]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    correct_letter = GPQA_LETTERS[order.index(0)]
    lines = [str(ex["Question"]).strip(), ""]
    lines += [f"({letter}) {text}" for letter, text in zip(GPQA_LETTERS, shuffled, strict=True)]
    lines += ["", "Answer with the letter of the correct option."]
    return "\n".join(lines), correct_letter


def rows_from_hf(name: str, dataset: Iterable[dict]) -> Iterator[Row]:
    """Convert one benchmark's HF rows to :class:`Row`."""
    if name == DS_MATH500:
        for i, ex in enumerate(dataset):
            yield Row(
                DS_MATH500,
                ex["problem"],
                str(ex["answer"]),
                "test",
                str(ex.get("unique_id", i)),
                extra={"subject": ex.get("subject"), "level": ex.get("level")},
            )
    elif name == DS_AIME24:
        for i, ex in enumerate(dataset):
            answer = last_boxed_content(str(ex["solution"])) or str(ex["solution"]).strip()
            yield Row(
                DS_AIME24, ex["problem"], answer, "test", str(ex.get("id", i)), extra={"url": ex.get("url")}
            )
    elif name == DS_AIME25:
        for i, ex in enumerate(dataset):
            yield Row(DS_AIME25, ex["problem"], str(ex["answer"]), "test", str(ex.get("id", i)))
    elif name == DS_AMC23:
        for i, ex in enumerate(dataset):
            yield Row(
                DS_AMC23,
                ex["question"],
                str(ex["answer"]),
                "test",
                str(ex.get("id", i)),
                extra={"url": ex.get("url")},
            )
    elif name == DS_GPQA:
        for i, ex in enumerate(dataset):
            rng = random.Random(f"gpqa-{i}")
            question, letter = _gpqa_question(ex, rng)
            yield Row(DS_GPQA, question, letter, "test", str(ex.get("Record ID", i)), ability="science")
    else:
        raise ValueError(f"unknown benchmark {name!r}; known: {sorted(HF_SPECS)}")
