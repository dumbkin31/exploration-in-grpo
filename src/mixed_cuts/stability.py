"""fp16 stability watchlist (Task D).

RL fine-tuning in fp16 on sm_75 is meaningfully less stable than the bf16 setups the papers used,
and "it did not crash" is not the bar. After every training step :class:`StabilityWatch` inspects
verl's actor metrics and raises loud alerts (log banners + ``stability/alert_*`` = 1 metrics) for:

- NaN or inf in any ``actor/*`` metric (losses, grad norm, entropy, KL);
- gradient-norm spikes: ``actor/grad_norm`` above an absolute ceiling, or above
  ``spike_factor`` times the running median of the previous steps;
- policy-entropy collapse in the first ``entropy_collapse_window`` steps: entropy below
  ``entropy_collapse_ratio`` times the step-1 entropy;
- policy drift, the onset of the H200 divergence (decisions 016, 018): ``|actor/ppo_kl|`` (the policy's
  drift from pi_old inside one step) above ``ppo_kl_max``, or entropy above ``entropy_rise_ratio``
  times its lowest value so far (after ``entropy_rise_after`` steps). With ``abort_on_divergence``,
  ``divergence_patience`` consecutive drift steps stop the run.

The drift thresholds come from the H200 logs (``results/h200-2026-10-01/training``): before diverging,
``|ppo_kl|`` stayed below 0.0096 in both arms and entropy never doubled from its minimum; ``ppo_kl``
crossed 0.02 at Mixed-CUTS step 49 and GRPO step 66, entropy doubled one step later, and both kept
climbing (``ppo_kl`` up to 1.5, entropy up to 6.8). Patience 2 stops them at steps 50 and 67.

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
PPO_KL_KEY = "actor/ppo_kl"


class StabilityAbort(MixedCutsStartupError):
    """Raised on NaN/inf (``abort_on_nan``) or a sustained policy drift (``abort_on_divergence``)."""


@dataclass(frozen=True)
class StabilityConfig:
    enabled: bool = True
    grad_norm_max: float = 50.0
    grad_norm_spike_factor: float = 10.0
    grad_norm_history: int = 20
    entropy_collapse_window: int = 20
    entropy_collapse_ratio: float = 0.25
    abort_on_nan: bool = False
    ppo_kl_max: float = 0.02
    entropy_rise_ratio: float = 2.0
    entropy_rise_after: int = 5
    divergence_patience: int = 2
    abort_on_divergence: bool = False

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
        self._entropy_min: float | None = None
        self._drift_streak = 0  # consecutive drift steps; restarts at 0 after a resume

    # ------------------------------------------------------------------ resume support
    def rebuild(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Restore the grad-norm history, the step-1 entropy and the lowest entropy from ``metrics.jsonl``."""
        for rec in sorted(records, key=lambda r: int(r.get("step", 0))):
            g = rec.get(GRAD_NORM_KEY)
            if g is not None and not _is_bad(g):
                self._push_grad_norm(float(g))
            e = rec.get(ENTROPY_KEY)
            if e is not None and not _is_bad(e):
                self._entropy_min = (
                    float(e) if self._entropy_min is None else min(self._entropy_min, float(e))
                )
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
            "stability/alert_policy_drift": 0.0,
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

        drift = self._policy_drift(metrics, step)
        if drift:
            out["stability/alert_policy_drift"] = 1.0
            problems.append("policy drift: " + "; ".join(drift))
            self._drift_streak += 1
        else:
            self._drift_streak = 0
        out["stability/drift_streak"] = float(self._drift_streak)

        out["stability/n_alerts"] = float(sum(v for k, v in out.items() if k.startswith("stability/alert_")))
        for p in problems:
            logger.error("=" * 88 + "\nSTABILITY ALERT at step %d: %s\n" + "=" * 88, step, p)
        if out["stability/alert_nan"] and self.config.abort_on_nan:
            raise StabilityAbort(f"NaN/inf in actor metrics at step {step}: {bad}")
        if self.config.abort_on_divergence and self._drift_streak >= self.config.divergence_patience:
            raise StabilityAbort(
                f"policy drift for {self._drift_streak} consecutive steps (step {step}): {'; '.join(drift)}. "
                "The kept checkpoints are in checkpoints_kept/ (decision 018)."
            )
        return out

    def _policy_drift(self, metrics: Mapping[str, Any], step: int) -> list[str]:
        """Reasons this step looks like the onset of a divergence (empty when it does not)."""
        reasons = []
        k = metrics.get(PPO_KL_KEY)
        if k is not None and not _is_bad(k) and abs(float(k)) > self.config.ppo_kl_max:
            reasons.append(f"|ppo_kl| {abs(float(k)):.3g} > {self.config.ppo_kl_max}")
        e = metrics.get(ENTROPY_KEY)
        if e is not None and not _is_bad(e):
            e = float(e)
            low = self._entropy_min
            if low is not None and low > 0 and step > self.config.entropy_rise_after:
                if e > self.config.entropy_rise_ratio * low:
                    reasons.append(
                        f"entropy {e:.4g} > {self.config.entropy_rise_ratio} x its minimum {low:.4g}"
                    )
            self._entropy_min = e if low is None else min(low, e)
        return reasons
