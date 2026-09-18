"""MATH (Hendrycks et al.) training set via ``DigitalLearningGmbH/MATH-lighteval``.

Columns: ``problem``, ``level``, ``solution``, ``type``. The ground truth is the content of the
last ``\\boxed{}`` in the reference solution (same extraction verl's ``examples/data_preprocess/
math_dataset.py`` uses).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from mc_data.schema import DS_MATH_TRAIN, Row
from mixed_cuts.boxed import last_boxed_content

HF_NAME = "DigitalLearningGmbH/MATH-lighteval"


def rows_from_hf(dataset: Iterable[dict], split: str) -> Iterator[Row]:
    """Convert HF rows (any iterable of dicts) to :class:`Row`; skips rows with no boxed answer."""
    for i, ex in enumerate(dataset):
        answer = last_boxed_content(ex["solution"])
        if answer is None:
            continue
        yield Row(
            data_source=DS_MATH_TRAIN,
            question=ex["problem"],
            ground_truth=answer,
            split=split,
            index=f"{split}-{i}",
            extra={"level": ex.get("level"), "type": ex.get("type")},
        )
