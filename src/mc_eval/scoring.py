"""Pure metric code for the eval harness (no vLLM, no GPU).

Per problem we have ``n`` sampled responses, each with a 0/1 correctness (from the same reward
function used in training) and an extracted answer string. From those:

- ``pass@1``  mean correctness over the n samples (NOT a single greedy decode; AIME with 30
              problems is far too high-variance for that);
- ``pass@k``  the unbiased estimator of Chen et al. (2021): 1 - C(n-c, k)/C(n, k);
- ``maj@k``   1 if the majority answer among the first k samples is correct. Answers are
              clustered by a cheap string normalisation first, then clusters are merged with the
              math-verify equivalence check so ``\\frac{1}{2}`` and ``0.5`` vote together.

Benchmark-level numbers are means over problems with a bootstrap 95% CI over problems.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

EqualFn = Callable[[str, str], bool]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021). ``n`` samples, ``c`` correct, ``k <= n``."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


_STRIP_PATTERNS = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\[dt]frac"), r"\\frac"),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\!|\\,|\;|\\:|\\quad"), ""),
    (re.compile(r"\^\{\\circ\}|\^\\circ|°"), ""),
    (re.compile(r"\\%|%"), ""),
    (re.compile(r"\$"), ""),
    (re.compile(r"\s+"), ""),
]


def normalize_answer_string(answer: str | None) -> str:
    """Cheap canonical form for clustering identical answers before the expensive check."""
    if answer is None:
        return ""
    s = answer.strip()
    for pat, rep in _STRIP_PATTERNS:
        s = pat.sub(rep, s)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    return s


def cluster_answers(answers: Sequence[str | None], equal: EqualFn | None = None) -> list[list[int]]:
    """Group sample indices by equivalent answer. Empty answers are never merged with anything."""
    clusters: list[list[int]] = []
    reps: list[str] = []  # representative (raw) answer per cluster
    norm_reps: list[str] = []
    for i, a in enumerate(answers):
        if a is None or not str(a).strip():
            clusters.append([i])
            reps.append("")
            norm_reps.append("")
            continue
        na = normalize_answer_string(a)
        placed = False
        for ci, (rep, nrep) in enumerate(zip(reps, norm_reps, strict=True)):
            if not rep:
                continue
            if na == nrep or (equal is not None and equal(rep, a)):
                clusters[ci].append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(str(a))
            norm_reps.append(na)
    return clusters


def majority_answer(answers: Sequence[str | None], equal: EqualFn | None = None) -> tuple[str | None, int]:
    """Return (representative answer of the largest cluster, its size). Ties: earliest cluster."""
    clusters = cluster_answers(answers, equal)
    best = max(clusters, key=len)
    rep = answers[best[0]]
    return (None if rep is None or not str(rep).strip() else str(rep)), len(best)


def summarize_problem(
    correct: Sequence[float],
    answers: Sequence[str | None],
    ground_truth: str,
    *,
    k_values: Sequence[int],
    is_correct_answer: Callable[[str], bool],
    equal: EqualFn | None = None,
) -> dict[str, float]:
    """Per-problem metrics from n samples.

    Args:
        correct: 0/1 per sample.
        answers: extracted answer per sample (may be None).
        ground_truth: reference answer (only used for bookkeeping).
        k_values: ks for pass@k and maj@k (each <= n).
        is_correct_answer: scorer for the majority answer against the ground truth.
        equal: optional expensive equivalence used to merge answer clusters.
    """
    n = len(correct)
    c = int(round(float(sum(correct))))
    out: dict[str, float] = {"n": float(n), "n_correct": float(c), "pass@1": c / n if n else 0.0}
    for k in k_values:
        if k > n:
            continue
        out[f"pass@{k}"] = pass_at_k(n, c, k)
        maj, size = majority_answer(list(answers[:k]), equal)
        out[f"maj@{k}"] = 1.0 if (maj is not None and is_correct_answer(maj)) else 0.0
        out[f"maj@{k}_agreement"] = size / k
    out["frac_no_answer"] = sum(1 for a in answers if a is None or not str(a).strip()) / n if n else 0.0
    return out


def aggregate(
    per_problem: Sequence[dict[str, float]], *, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Mean over problems + bootstrap 95% CI (over problems) for every metric key."""
    if not per_problem:
        return {"n_problems": 0}
    keys = sorted({k for p in per_problem for k in p if k not in ("n", "n_correct")})
    rng = np.random.default_rng(seed)
    m = len(per_problem)
    out: dict[str, Any] = {"n_problems": m, "n_samples": int(per_problem[0].get("n", 0))}
    idx = rng.integers(0, m, size=(n_boot, m)) if m > 1 else None
    for k in keys:
        vals = np.array([p.get(k, np.nan) for p in per_problem], dtype=np.float64)
        out[k] = float(np.nanmean(vals))
        if idx is not None:
            boots = np.nanmean(vals[idx], axis=1)
            out[f"{k}_ci95"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
        out[f"{k}_std_over_problems"] = float(np.nanstd(vals))
    return out
