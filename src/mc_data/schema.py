"""The parquet row format verl's ``RLHFDataset`` expects, and the prompt convention we use.

One row per problem::

    data_source   str                       routes the reward function / eval metrics
    prompt        list[{"role","content"}]  chat messages (system prompt is NOT included here)
    ability       str                       "math" | "science"
    reward_model  {"style": "rule", "ground_truth": str}
    extra_info    {"split": str, "index": str, ...}   anything else, kept for provenance

verl reads ``data.prompt_key = prompt`` and ``data.reward_fn_key = data_source``. The same
instruction is used for training and evaluation so extraction (last ``\\boxed{}``) is consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Qwen's recommended math instruction. Every prompt ends with it (train and eval alike).
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# data_source values (also the benchmark names used by the eval harness)
DS_MATH_TRAIN = "DigitalLearningGmbH/MATH-lighteval"
DS_DAPO = "math_dapo"
DS_MATH500 = "math500"
DS_AIME24 = "aime24"
DS_AIME25 = "aime25"
DS_AMC23 = "amc23"
DS_GPQA = "gpqa_diamond"
EVAL_SOURCES = (DS_MATH500, DS_AIME24, DS_AIME25, DS_AMC23, DS_GPQA)


@dataclass
class Row:
    data_source: str
    question: str
    ground_truth: str
    split: str
    index: str
    ability: str = "math"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "data_source": self.data_source,
            "prompt": [{"role": "user", "content": f"{self.question.strip()}\n\n{BOXED_INSTRUCTION}"}],
            "ability": self.ability,
            "reward_model": {"style": "rule", "ground_truth": str(self.ground_truth)},
            "extra_info": {"split": self.split, "index": str(self.index), **self.extra},
        }
