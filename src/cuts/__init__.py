"""CUTS: Constrained Uniform Top-K Sampling (Liang et al., 2026), as a decoding-time operator.

This package is deliberately independent of verl and vLLM so it can be unit-tested on a
laptop or the login node without a GPU:

- :mod:`cuts.config`   -- ``CutsParams``: the hyper-parameters and how they travel inside
                          vLLM's per-request ``SamplingParams.extra_args``.
- :mod:`cuts.operator` -- ``cuts_transform``: the batched SELECT / FILTER / EQUALIZE step.
- :mod:`cuts.state`    -- ``CutsBatchState``: per-request bookkeeping (warm-up, |S_t| stats)
                          that mirrors vLLM's ``BatchUpdate`` contract.
- :mod:`cuts.stats`    -- JSONL writer/reader and per-step summary of the |S_t| statistics.
- :mod:`cuts.vllm_logits_processor` -- the thin vLLM ``LogitsProcessor`` wrapper (imports
                          vLLM lazily; everything else works without it).
"""

from cuts.config import EXTRA_ARGS_KEY, CutsParams
from cuts.operator import CutsResult, cuts_transform
from cuts.state import CutsBatchState, RequestEntry
from cuts.stats import CutsStatsWriter, read_cuts_stats, summarize_cuts_stats

__all__ = [
    "EXTRA_ARGS_KEY",
    "CutsParams",
    "CutsResult",
    "cuts_transform",
    "CutsBatchState",
    "RequestEntry",
    "CutsStatsWriter",
    "read_cuts_stats",
    "summarize_cuts_stats",
]
