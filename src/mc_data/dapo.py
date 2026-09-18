"""DAPO-Math-17k (``BytedTsinghua-SIA/DAPO-Math-17k``).

The Hub parquet has **1,791,700 rows for ~17.9k problems**: every prompt is repeated ~100x for
compatibility with an old verl version that expected pre-duplicated rows. Anything that computes
per-prompt statistics would be wrong on the raw file, so :func:`deduplicate` runs first and the
row count is asserted to have shrunk by roughly that factor.

Rows already follow verl's schema (``data_source: math_dapo``, ``prompt``, ``reward_model``,
``extra_info.index`` = a UUID per problem). DAPO's own prompt asks for a last line of the form
``Answer: $Answer``; we replace that wrapper with our uniform ``\\boxed{}`` instruction so train
and eval extraction match. The original prompt text is kept in ``extra_info.original_prompt``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from mc_data.schema import DS_DAPO, Row

HF_NAME = "BytedTsinghua-SIA/DAPO-Math-17k"

# DAPO wraps every problem in this preamble / postamble (verified on the Hub's first rows).
_PREAMBLE = re.compile(
    r"^\s*Solve the following math problem step by step\. The last line of your response should be of the form "
    r"Answer: \$Answer \(without quotes\) where \$Answer is the answer to the problem\.\s*",
    re.DOTALL,
)
_POSTAMBLE = re.compile(r"\s*Remember to put your answer on its own line after \"Answer:\"\.?\s*$", re.DOTALL)


def strip_dapo_wrapper(content: str) -> str:
    content = _PREAMBLE.sub("", content)
    content = _POSTAMBLE.sub("", content)
    return content.strip()


def _key(ex: dict) -> str:
    idx = (ex.get("extra_info") or {}).get("index")
    if idx is not None:
        return str(idx)
    # fallback: hash of the prompt text
    prompt = ex.get("prompt") or []
    return "prompt:" + str(hash(tuple((m.get("role"), m.get("content")) for m in prompt)))


def deduplicate(dataset: Iterable[dict]) -> tuple[list[dict], int]:
    """Return (unique rows, total rows seen). First occurrence wins."""
    seen: set[str] = set()
    unique: list[dict] = []
    total = 0
    for ex in dataset:
        total += 1
        k = _key(ex)
        if k in seen:
            continue
        seen.add(k)
        unique.append(ex)
    return unique, total


def rows_from_hf(dataset: Iterable[dict], split: str = "train") -> Iterator[Row]:
    for ex in dataset:
        messages = ex["prompt"]
        original = messages[-1]["content"] if messages else ""
        gt = str((ex.get("reward_model") or {}).get("ground_truth", ""))
        idx = str((ex.get("extra_info") or {}).get("index"))
        yield Row(
            data_source=DS_DAPO,
            question=strip_dapo_wrapper(original),
            ground_truth=gt,
            split=split,
            index=idx,
            extra={"original_prompt": original},
        )
