"""fp16 stability watchlist (Task D).

RL fine-tuning in fp16 on sm_75 is meaningfully less stable than the bf16 setups the papers used,
and "it did not crash" is not the bar. After every training step :class:`StabilityWatch` inspects
verl's actor metrics and raises loud alerts (log banners + ``stability/alert_*`` = 1 metrics) for:

- NaN or inf in any ``actor/*`` metric (losses, grad norm, entropy, KL);
- gradient-norm spikes: ``actor/grad_norm`` above an absolute ceiling, or above
  ``spike_factor`` times the running median of the previous steps;
- policy-entropy collapse in the first ``entropy_collapse_window`` steps: entropy below
  ``entropy_collapse_ratio`` times the step-1 entropy.

The mitigation ladder (docs/decisions/006-fp16-stability-ladder.md): lower LR toward 5e-7,
tighten ``actor.grad_clip``, then reconsider the KL term; whichever rung is used is a recorded
deviation from the paper. ``abort_on_nan`` turns the NaN alert into an exception.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from mixed_cuts.errors import MixedCutsStartupError

logger = logging.getLogger(__name__)

GRAD_NORM_KEY = "actor/grad_norm"
ENTROPY_KEY = "actor/entropy"


class StabilityAbort(MixedCutsStartupError):
    """Raised when ``abort_on_nan`` is set and a NaN/inf shows up in the actor metrics."""


@dataclass(frozen=True)
class StabilityConfig:
    enabled: bool = True
    grad_norm_max: float = 50.0
    grad_norm_spike_factor: float = 10.0
    grad_norm_history: int = 20
    entropy_collapse_window: int = 20
    entropy_collapse_ratio: float = 0.25
    abort_on_nan: bool = False

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> StabilityConfig:
        if cfg is None:
            return cls()
        return cls(**{k: cfg[k] for k in cfg.keys()})


def _is_bad(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isnan(v) or math.isinf(v)


class StabilityWatch:
    def __init__(self, config: StabilityConfig) -> None:
        self.config = config
        self._grad_norms: list[float] = []
        self._entropy_ref: float | None = None

    # ------------------------------------------------------------------ resume support
    def rebuild(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Restore the grad-norm history and the step-1 entropy from ``metrics.jsonl`` records."""
        for rec in sorted(records, key=lambda r: int(r.get("step", 0))):
            g = rec.get(GRAD_NORM_KEY)
            if g is not None and not _is_bad(g):
                self._push_grad_norm(float(g))
            if (
                int(rec.get("step", 0)) == 1
                and rec.get(ENTROPY_KEY) is not None
                and not _is_bad(rec[ENTROPY_KEY])
            ):
                self._entropy_ref = float(rec[ENTROPY_KEY])

    def _push_grad_norm(self, value: float) -> None:
        self._grad_norms.append(value)
        if len(self._grad_norms) > self.config.grad_norm_history:
            self._grad_norms.pop(0)

    # ------------------------------------------------------------------ the check
    def check(self, metrics: Mapping[str, Any], step: int) -> dict[str, float]:
        """Return ``stability/*`` metrics for this step; log banners for every alert."""
        out = {
            "stability/alert_nan": 0.0,
            "stability/alert_grad_spike": 0.0,
            "stability/alert_entropy_collapse": 0.0,
        }
        if not self.config.enabled:
            return out
        problems: list[str] = []

        bad = sorted(k for k, v in metrics.items() if k.startswith("actor/") and _is_bad(v))
        if bad:
            out["stability/alert_nan"] = 1.0
            problems.append(f"NaN/inf in {bad}")

        g = metrics.get(GRAD_NORM_KEY)
        if g is not None and not _is_bad(g):
            g = float(g)
            ref = median(self._grad_norms) if len(self._grad_norms) >= 5 else None
            if g > self.config.grad_norm_max:
                out["stability/alert_grad_spike"] = 1.0
                problems.append(f"grad_norm {g:.3g} > grad_norm_max {self.config.grad_norm_max}")
            elif ref is not None and ref > 0 and g > self.config.grad_norm_spike_factor * ref:
                out["stability/alert_grad_spike"] = 1.0
                problems.append(
                    f"grad_norm {g:.3g} > {self.config.grad_norm_spike_factor}x running median {ref:.3g}"
                )
            self._push_grad_norm(g)
            if ref is not None:
                out["stability/grad_norm_over_median"] = g / ref if ref > 0 else float("nan")

        e = metrics.get(ENTROPY_KEY)
        if e is not None and not _is_bad(e):
            e = float(e)
            if step == 1 or self._entropy_ref is None:
                self._entropy_ref = e
            elif (
                step <= self.config.entropy_collapse_window
                and e < self.config.entropy_collapse_ratio * self._entropy_ref
            ):
                out["stability/alert_entropy_collapse"] = 1.0
                problems.append(
                    f"entropy {e:.4g} < {self.config.entropy_collapse_ratio} x step-1 entropy {self._entropy_ref:.4g} "
                    f"(step {step} <= {self.config.entropy_collapse_window})"
                )
            out["stability/entropy_over_step1"] = e / self._entropy_ref if self._entropy_ref else float("nan")

        out["stability/n_alerts"] = float(sum(v for k, v in out.items() if k.startswith("stability/alert_")))
        for p in problems:
            logger.error("=" * 88 + "\nSTABILITY ALERT at step %d: %s\n" + "=" * 88, step, p)
        if out["stability/alert_nan"] and self.config.abort_on_nan:
            raise StabilityAbort(f"NaN/inf in actor metrics at step {step}: {bad}")
        return out
