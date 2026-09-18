"""Rule-based reward: last ``\\boxed{}`` extraction + math-verify equivalence.

Used by verl through ``reward.custom_reward_function`` (signature fixed by
``verl/experimental/reward_loop/reward_manager/naive.py``)::

    compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs)

and by the eval harness, so training reward and evaluation correctness are the same function.

Returns a dict; verl takes ``"score"`` as the reward and stores the other keys as
``reward_extra_info`` (they must be present for every sample, so keys are constant):

    score      1.0 / 0.0
    pred       the extracted answer string ("" if nothing was extracted)
    has_boxed  1.0 if a ``\\boxed{}`` was found, else 0.0   (format diagnostic)

math-verify uses ``signal.alarm`` for its internal timeout, which only works in the main thread;
verl calls the reward from a thread-pool executor, so like verl's own ``math_verify.py`` we run
the check in a spawned subprocess pool with a hard timeout (timeouts score 0).
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from mc_data.schema import DS_GPQA
from mixed_cuts.boxed import last_answer_line, last_boxed_content, normalize_letter

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
    return bool(verify(gold_parsed, pred_parsed))


def extract_answer(solution_str: str) -> tuple[str | None, bool]:
    """Return (answer, found_boxed). Falls back to an ``Answer:`` line when no box exists."""
    boxed = last_boxed_content(solution_str)
    if boxed is not None:
        return boxed.strip(), True
    line = last_answer_line(solution_str)
    return (line, False) if line else (None, False)


def score_math(
    solution_str: str, ground_truth: str, *, timeout: float = 30.0, require_boxed: bool = False
) -> dict[str, Any]:
    pred, has_boxed = extract_answer(solution_str)
    result = {"score": 0.0, "pred": pred or "", "has_boxed": float(has_boxed)}
    if pred is None or (require_boxed and not has_boxed):
        return result
    gt = str(ground_truth).strip()
    if pred == gt:  # exact match short-circuit (also avoids a subprocess round trip)
        result["score"] = 1.0
        return result
    try:
        equal = _get_pool().submit(_math_verify_equal, gt, pred).result(timeout=timeout)
    except FuturesTimeoutError:
        equal = False
    except Exception:  # noqa: BLE001 - a parser crash is a wrong answer, not a crashed run
        equal = False
    result["score"] = 1.0 if equal else 0.0
    return result


def score_multiple_choice(solution_str: str, ground_truth: str) -> dict[str, Any]:
    pred, has_boxed = extract_answer(solution_str)
    letter = normalize_letter(pred)
    return {
        "score": 1.0 if letter is not None and letter == str(ground_truth).strip().upper() else 0.0,
        "pred": letter or (pred or ""),
        "has_boxed": float(has_boxed),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,  # noqa: ARG001 - verl passes it; unused
    *,
    timeout: float = 30.0,
    require_boxed: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """verl entry point (see module docstring)."""
    if data_source == DS_GPQA:
        return score_multiple_choice(solution_str, ground_truth)
    return score_math(solution_str, ground_truth, timeout=timeout, require_boxed=require_boxed)


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
