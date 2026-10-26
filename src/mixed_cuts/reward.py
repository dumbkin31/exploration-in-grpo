"""Rule-based reward (CUTS paper Appendix B.4): final ``\\boxed{}`` + math-verify equivalence.

Used by verl through ``reward.custom_reward_function`` (signature fixed by
``verl/experimental/reward_loop/reward_manager/naive.py``)::

    compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs)

and by the eval harness, so training reward and evaluation correctness are the same function.

Math benchmarks (MATH, MATH-500, AIME, AMC): a response is VALID if a ``\\boxed{...}`` is present
and the content of the FINAL one is not empty; the content (any LaTeX: ``\\frac{3}{4}``,
``2\\sqrt{5}``, ``\\pi``, ``x^2+1``) is compared with the ground truth by math-verify symbolic
equivalence. Binary reward; an answer math-verify cannot parse, or that times out, scores 0. There
is no ``Answer:``-line fallback on this path. Until decision 017 the content also had to contain a
digit (the paper's Appendix B.4 rule, decision 008), so ground truths such as ``\\pi`` could never be
rewarded; ``scripts/prepare_data.py`` still drops those from the training set (unchanged, for
comparability with the earlier runs).

GPQA (multiple choice) uses the separate letter parser in :mod:`mixed_cuts.gpqa`.

Returns a dict; verl takes ``"score"`` as the reward and stores the other keys as
``reward_extra_info`` (they must be present for every sample, so the keys are constant):

    score      1.0 / 0.0
    pred       the extracted answer string ("" if nothing was extracted)
    has_boxed  1.0 if a \\boxed{} was found, else 0.0        (format diagnostic)
    valid      1.0 if the response passed the validity rule  (non-empty final box, or a letter for GPQA)

math-verify uses ``signal.alarm`` for its internal timeouts (5 s per parse, 5 s per gold x pred
comparison), which only works in the main thread; verl calls the reward from worker threads, so
like verl's own ``math_verify.py`` we run the check in a spawned subprocess pool of
``MC_REWARD_WORKERS`` processes (one pool per process that imports this module).

How verl calls us (verl @ 483b8a0): each of the ``reward.num_workers`` (default 8) RewardLoopWorker
Ray actors is an async actor (Ray's default max_concurrency is 1000) and runs every
``compute_score`` through ``loop.run_in_executor(None, ...)`` (reward_manager/naive.py:75), i.e.
asyncio's default thread pool of min(32, os.cpu_count() + 4) threads (Python 3.12; os.cpu_count()
is the whole node, not the job's allocation). Up to 32 calls per process therefore share a few pool
workers. A call first waits for an idle worker and only then starts its
``timeout``, so the timeout bounds the comparison's own run time, not the wait behind other answers.

Pool failures are logged and never silently turn the reward into exact string matching:

- a worker that dies (OOM, segfault, killed) breaks the executor for good: the pool is replaced
  and the comparison retried once;
- a comparison still running ``timeout`` s after it started is stuck where math-verify's alarm
  cannot interrupt it (C code): it scores 0 and the pool's workers are terminated and replaced
  (comparisons that were running beside it are retried once);
- any other exception from math-verify scores 0 (a parser crash is a wrong answer, not a crashed run);
- ``MC_REWARD_WORKERS`` < 1 raises instead of grading without math-verify.

``pool_restarts()`` counts the pools replaced in this process.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from mc_data.schema import DS_GPQA
from mixed_cuts.boxed import last_boxed_content
from mixed_cuts.gpqa import extract_choice_letter

logger = logging.getLogger(__name__)


class _VerifyPool:
    """A spawn-context ProcessPoolExecutor whose ``submit`` waits until a worker is idle.

    ``idle`` holds one permit per worker that is not running a comparison. A caller takes a permit
    before submitting and gets it back when the comparison finishes (also if the caller has already
    given up on it), so a submitted comparison starts right away instead of queueing in the executor.
    """

    def __init__(self, workers: int):
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.idle = threading.BoundedSemaphore(workers)

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        self.idle.acquire()
        try:
            future = self.executor.submit(fn, *args)
        except BaseException:
            self.idle.release()
            raise
        future.add_done_callback(lambda _: self.idle.release())
        return future

    def terminate_workers(self) -> None:
        """Kill the worker processes; the executor then fails the in-flight futures (BrokenProcessPool).

        No public API for this before Python 3.14 (``ProcessPoolExecutor.terminate_workers``);
        ``_processes`` is CPython's {pid: Process} map (concurrent/futures/process.py, 3.10-3.13).
        """
        for proc in list((self.executor._processes or {}).values()):
            proc.terminate()


_pool: _VerifyPool | None = None
_pool_lock = threading.Lock()
_pool_restarts = 0


def _get_pool() -> _VerifyPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                workers = int(os.environ.get("MC_REWARD_WORKERS", "4"))
                if workers < 1:
                    # without a pool every answer that is not an exact string match would score 0
                    raise ValueError(f"MC_REWARD_WORKERS must be >= 1 (got {workers})")
                _pool = _VerifyPool(workers)
    return _pool


def _discard_pool(pool: _VerifyPool, why: str, *, terminate: bool = False) -> None:
    """Make the next call build a fresh pool. Only the first caller acts; the rest find it replaced."""
    global _pool, _pool_restarts
    with _pool_lock:
        if _pool is not pool:
            return
        _pool = None
        _pool_restarts += 1
        restarts = _pool_restarts
    logger.warning("math-verify pool replaced: %s (%d replacement(s) in this process)", why, restarts)
    if terminate:
        # no shutdown(): the executor marks itself broken once it sees the dead workers, so a thread
        # still holding this pool gets BrokenProcessPool (and retries) rather than a RuntimeError
        pool.terminate_workers()
    else:
        pool.executor.shutdown(wait=False, cancel_futures=True)


def pool_restarts() -> int:
    """Number of math-verify pools replaced in this process (a worker died or got stuck)."""
    return _pool_restarts


def _math_verify_equal(gold: str, pred: str) -> bool:
    """Runs inside the subprocess. Both sides are wrapped in \\boxed{} so LaTeX extraction applies."""
    from math_verify.grader import verify
    from math_verify.parser import LatexExtractionConfig, parse

    gold_parsed = parse(f"\\boxed{{{gold}}}", (LatexExtractionConfig(),))
    pred_parsed = parse(f"\\boxed{{{pred}}}", (LatexExtractionConfig(),))
    if not gold_parsed or not pred_parsed:
        return False
    return bool(verify(gold_parsed, pred_parsed))


def _verify_in_pool(gold: str, pred: str, timeout: float) -> bool:
    """math-verify equivalence in the worker pool (failure handling: see the module docstring)."""
    for _attempt in range(2):  # one retry when the pool dies under us
        pool = _get_pool()
        try:
            future = pool.submit(_math_verify_equal, gold, pred)  # waits for an idle worker
            return bool(future.result(timeout=timeout))
        except BrokenProcessPool:
            _discard_pool(pool, "a worker process died")
        except FuturesTimeoutError:
            _discard_pool(pool, f"a comparison was still running after {timeout:g} s", terminate=True)
            logger.warning("math-verify timed out; scoring 0 (pred=%.80r)", pred)
            return False
        except Exception as e:  # noqa: BLE001 - a parser crash is a wrong answer, not a crashed run
            logger.warning("math-verify raised %s; scoring 0 (pred=%.80r)", type(e).__name__, pred)
            return False
    logger.warning("math-verify pool died on the retry as well; scoring 0 (pred=%.80r)", pred)
    return False


def has_digit(text: str | None) -> bool:
    """The training-set filter of decision 008 (scripts/prepare_data.py); no longer part of the reward."""
    return bool(text) and any(ch.isdigit() for ch in text)


def extract_answer(solution_str: str) -> tuple[str | None, bool, bool]:
    """Return (final boxed content or None, has_boxed, valid). valid = boxed AND non-empty (decision 017)."""
    boxed = last_boxed_content(solution_str)
    if boxed is None:
        return None, False, False
    pred = boxed.strip()
    return pred, True, len(pred) > 0


def score_math(solution_str: str, ground_truth: str, *, timeout: float = 30.0) -> dict[str, Any]:
    pred, has_boxed, valid = extract_answer(solution_str)
    result = {"score": 0.0, "pred": pred or "", "has_boxed": float(has_boxed), "valid": float(valid)}
    if not valid:
        return result
    gt = str(ground_truth).strip()
    if pred == gt:  # exact match short-circuit (also avoids a subprocess round trip)
        result["score"] = 1.0
        return result
    result["score"] = 1.0 if _verify_in_pool(gt, pred, timeout) else 0.0
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
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.executor.shutdown(wait=False, cancel_futures=True)
