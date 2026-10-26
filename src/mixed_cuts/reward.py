"""Rule-based reward (CUTS paper Appendix B.4): final ``\\boxed{}`` + math-verify equivalence.

Used by verl through ``reward.custom_reward_function`` (signature fixed by
``verl/experimental/reward_loop/reward_manager/naive.py``)::

    compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs)

and by the eval harness, so training reward and evaluation correctness are the same function.

Math benchmarks (MATH, MATH-500, AIME, AMC): a response is VALID only if a ``\\boxed{...}`` is
present and the content of the FINAL one contains at least one numeric digit; the content is
compared with the ground truth by math-verify symbolic equivalence. Binary reward. There is no
``Answer:``-line fallback on this path. Ground truths without any digit (e.g. ``\\pi``) can never
be rewarded under this rule; ``scripts/prepare_data.py`` filters them out of the training set and
the eval harness reports the resulting ceiling per benchmark (docs/decisions/008).

Equivalence is by mathematical value, not text: ``1/2``, ``0.5``, ``.5``, ``0.50``, ``\\frac{1}{2}``,
``\\dfrac12`` and ``2/4`` all match each other, ``0.25 == 1/4``, ``1,000 == 1000.0`` and
``\\sqrt{20} == 2\\sqrt{5}``. math-verify parses decimals to floats, so any comparison with a
decimal goes through :data:`FLOAT_ROUNDING` (see there). The one custom normalization is a percent
sign present on BOTH sides (:func:`_strip_shared_percent`); whether ``50\\%`` should match ``50``
or ``0.5`` is an open policy question and stays a mismatch.

GPQA (multiple choice) uses the separate letter parser in :mod:`mixed_cuts.gpqa`.

Returns a dict; verl takes ``"score"`` as the reward and stores the other keys as
``reward_extra_info`` (they must be present for every sample, so the keys are constant):

    score      1.0 / 0.0
    pred       the extracted answer string ("" if nothing was extracted)
    has_boxed  1.0 if a \\boxed{} was found, else 0.0        (format diagnostic)
    valid      1.0 if the response passed the validity rule  (boxed + digit, or a letter for GPQA)

math-verify uses ``signal.alarm`` for its internal timeout, which only works in the main thread;
verl calls the reward from a thread-pool executor, so like verl's own ``math_verify.py`` we run
the check in a spawned subprocess pool with a hard timeout (timeouts score 0).
"""

from __future__ import annotations

import multiprocessing
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from mc_data.schema import DS_GPQA
from mixed_cuts.boxed import last_boxed_content
from mixed_cuts.gpqa import extract_choice_letter

# math-verify's numeric tolerance, passed explicitly (these ARE its defaults, math_verify/grader.py
# ``verify``). When either side is a decimal (sympy Float), both sides are rounded to
# FLOAT_ROUNDING decimal places and compared exactly (``sympy_numeric_eq``): 1/3 matches 0.333333
# but not 0.333 or 0.33, and 2\sqrt{5} does not match 4.47. It applies to every decimal, so 1/4
# also matches 0.2500001, and it is ABSOLUTE, so tiny values get less room (gold .0000672 also
# matches 0.0000669). Changing it is a policy decision.
FLOAT_ROUNDING = 6
NUMERIC_PRECISION = 15

# A trailing percent sign: "12.5\%", "12.5 %", "12.5\ \%".
_PERCENT_SUFFIX = re.compile(r"(?:\\[ ,;!]|\s)*\\?%\s*$")

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                workers = int(os.environ.get("MC_REWARD_WORKERS", "4"))
                _pool = ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                )
    return _pool


def _math_verify_equal(gold: str, pred: str) -> bool:
    """Runs inside the subprocess. Both sides are wrapped in \\boxed{} so LaTeX extraction applies."""
    from math_verify.grader import verify
    from math_verify.parser import LatexExtractionConfig, parse

    gold_parsed = parse(f"\\boxed{{{gold}}}", (LatexExtractionConfig(),))
    pred_parsed = parse(f"\\boxed{{{pred}}}", (LatexExtractionConfig(),))
    if not gold_parsed or not pred_parsed:
        return False
    return bool(
        verify(gold_parsed, pred_parsed, float_rounding=FLOAT_ROUNDING, numeric_precision=NUMERIC_PRECISION)
    )


def _strip_shared_percent(gold: str, pred: str) -> tuple[str, str]:
    """Drop a trailing percent sign when BOTH answers carry one, so ``12.5\\%`` matches ``12.50\\%``.

    math-verify cannot parse percentages in our environment: verl's hydra/omegaconf pin
    antlr4-python3-runtime 4.9.3, and latex2sympy2_extended 1.11.0's parser generated for ANTLR
    4.9.3 lacks the ``FUNC_GAMMA`` token that ``convert_atom`` checks before ``PERCENT_NUMBER``.
    Every percent atom raises AttributeError, ``parse`` falls back to the raw string, and
    ``verify`` only matches identical strings. Answers with the same unit on both sides are
    compared as plain numbers; a percent on one side only is left alone (still a mismatch).
    """
    g, p = _PERCENT_SUFFIX.sub("", gold), _PERCENT_SUFFIX.sub("", pred)
    if g != gold and p != pred and "%" not in g and "%" not in p:
        return g, p
    return gold, pred


def has_digit(text: str | None) -> bool:
    return bool(text) and any(ch.isdigit() for ch in text)


def extract_answer(solution_str: str) -> tuple[str | None, bool, bool]:
    """Return (final boxed content or None, has_boxed, valid). valid = boxed AND contains a digit."""
    boxed = last_boxed_content(solution_str)
    if boxed is None:
        return None, False, False
    pred = boxed.strip()
    return pred, True, has_digit(pred)


def score_math(solution_str: str, ground_truth: str, *, timeout: float = 30.0) -> dict[str, Any]:
    pred, has_boxed, valid = extract_answer(solution_str)
    result = {"score": 0.0, "pred": pred or "", "has_boxed": float(has_boxed), "valid": float(valid)}
    if not valid:
        return result
    gt = str(ground_truth).strip()
    gt_cmp, pred_cmp = _strip_shared_percent(gt, pred)
    if pred_cmp == gt_cmp:  # exact match short-circuit (also avoids a subprocess round trip)
        result["score"] = 1.0
        return result
    try:
        equal = _get_pool().submit(_math_verify_equal, gt_cmp, pred_cmp).result(timeout=timeout)
    except FuturesTimeoutError:
        equal = False
    except Exception:  # noqa: BLE001 - a parser crash is a wrong answer, not a crashed run
        equal = False
    result["score"] = 1.0 if equal else 0.0
    return result


def score_multiple_choice(solution_str: str, ground_truth: str) -> dict[str, Any]:
    letter = extract_choice_letter(solution_str)
    return {
        "score": 1.0 if letter is not None and letter == str(ground_truth).strip().upper() else 0.0,
        "pred": letter or "",
        "has_boxed": float(last_boxed_content(solution_str) is not None),
        "valid": float(letter is not None),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,  # noqa: ARG001 - verl passes it; unused
    *,
    timeout: float = 30.0,
    **_: Any,
) -> dict[str, Any]:
    """verl entry point (see module docstring)."""
    if data_source == DS_GPQA:
        return score_multiple_choice(solution_str, ground_truth)
    return score_math(solution_str, ground_truth, timeout=timeout)


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
