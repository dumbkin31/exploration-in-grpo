"""Hyper-parameters of the CUTS operator and their per-request transport.

A CUTS request is any vLLM request whose ``SamplingParams.extra_args`` contains the key
``"cuts"`` (:data:`EXTRA_ARGS_KEY`) mapping to a dict with the fields of :class:`CutsParams`.
Standard requests simply do not carry the key, which is how standard and CUTS rollouts
coexist inside one vLLM batch.

Paper defaults (Liang et al., 2026, Table 3): K=5, delta=0.03, T_warm=5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

EXTRA_ARGS_KEY = "cuts"

# When no top-K candidate reaches ``delta``:
#   "topk"   -> sample uniformly from the whole top-K set (the paper's rule)
#   "argmax" -> keep only the single most likely token (greedy)
EMPTY_SET_FALLBACKS = ("topk", "argmax")


@dataclass(frozen=True)
class CutsParams:
    """Everything the logits processor needs to know about one CUTS request."""

    k: int = 5
    """SELECT: number of top candidates considered at each step."""
    delta: float = 0.03
    """FILTER: absolute probability threshold (computed at temperature 1.0, see README)."""
    t_warm: int = 5
    """PREFIX PROTECTION: number of generated tokens sampled normally before CUTS kicks in."""
    empty_set_fallback: str = "topk"
    """What to do when the filter leaves nothing; see :data:`EMPTY_SET_FALLBACKS`."""

    # --- bookkeeping only (optional); used to attribute |S_t| statistics to a rollout ---
    stats_dir: str | None = None
    """Directory where the logits processor appends per-request stats as JSONL. None = off."""
    step: int | None = None
    """Training global step that issued the request."""
    uid: str | None = None
    """verl prompt uid (group id)."""
    session_id: int | None = None
    """Index of the rollout inside its group."""

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"cuts.k must be >= 1, got {self.k}")
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"cuts.delta must be in [0, 1], got {self.delta}")
        if self.t_warm < 0:
            raise ValueError(f"cuts.t_warm must be >= 0, got {self.t_warm}")
        if self.empty_set_fallback not in EMPTY_SET_FALLBACKS:
            raise ValueError(
                f"cuts.empty_set_fallback must be one of {EMPTY_SET_FALLBACKS}, got {self.empty_set_fallback!r}"
            )

    # --- transport ---------------------------------------------------------------------
    def to_extra_args(self) -> dict[str, Any]:
        """The ``extra_args`` dict to put on ``SamplingParams`` for a CUTS request."""
        return {EXTRA_ARGS_KEY: asdict(self)}

    @classmethod
    def from_extra_args(cls, extra_args: dict[str, Any] | None) -> CutsParams | None:
        """Parse ``SamplingParams.extra_args``; returns ``None`` for a standard request.

        Unknown keys are rejected so a typo in a config (``t_warm`` vs ``twarm``) fails loudly
        at request validation time instead of silently running standard sampling.
        """
        if not extra_args:
            return None
        raw = extra_args.get(EXTRA_ARGS_KEY)
        if raw is None:
            return None
        if isinstance(raw, CutsParams):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"extra_args[{EXTRA_ARGS_KEY!r}] must be a dict, got {type(raw).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown CUTS parameter(s) {sorted(unknown)}; known: {sorted(known)}")
        return cls(**raw)

    @property
    def signature(self) -> tuple[int, float, str]:
        """The subset of fields that changes the operator's maths (used to batch requests)."""
        return (self.k, self.delta, self.empty_set_fallback)
