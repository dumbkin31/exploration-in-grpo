"""Group-level reward diagnostics for Mixed-CUTS (pure numpy / torch, no verl).

Everything here answers *why* a run learns or does not, at the granularity GRPO cares about:
the prompt group. Inputs are flat per-rollout arrays (one entry per rollout) plus the group id
(``uid``) of each rollout and, optionally, its kind (``"std"`` / ``"cuts"``).

Metric glossary (README section 7). All population statistics use ``ddof=0``.

Advantage collapse (He et al., arXiv 2605.21125, Definition 4.1):
- ``mixed_cuts/advantage_collapse_rate``  ACR = (1/N) sum_j I(sigma_Rj < tau), the fraction of the
  N prompt groups whose G rewards have population std below tau = 1e-6. For those groups the
  normalised advantage is exactly zero, so they contribute no gradient.
- ``mixed_cuts/all_correct_frac`` / ``all_wrong_frac``  the two collapse modes reported
  separately (He et al. Table 3): every reward == 1, every reward == 0.
- ``mixed_cuts/std_subgroup_collapse_rate`` / ``cuts_subgroup_collapse_rate``  ACR of the standard
  / CUTS rollouts of each group taken alone (groups with that kind present).
- ``mixed_cuts/frac_groups_saved_by_cuts``  among groups that contain both kinds: the standard
  rollouts alone collapsed, but the mixed group did not. This is Mixed-CUTS's whole point.

Variance decomposition (Liang et al., Section 2.3, Eq. 5) for a group split evenly into
``G_std`` and ``G_CUTS``::

    sigma^2_mixed = 0.5 * (sigma^2_std + sigma^2_CUTS) + 0.25 * (mu_std - mu_CUTS)^2

- ``mixed_cuts/decomp/{mu_std, mu_cuts, var_std, var_cuts, between, sigma2_mixed}``  per-group
  terms averaged over the groups that contain both kinds; ``between`` = 0.25 (mu_std - mu_cuts)^2;
  ``sigma2_mixed`` is the measured population variance of the whole group.
- ``mixed_cuts/decomp/identity_residual``  mean |sigma2_mixed - rhs|; exactly 0 for an 8/8 split,
  non-zero when the split is unequal (the paper's identity assumes equal halves).

Cost accounting (He et al. Table 5):
- ``mixed_cuts/rollouts_generated_per_step``  number of (non-padding) rollouts this step.
- ``mixed_cuts/sample_utilization``  fraction of rollouts that contribute a non-zero gradient,
  i.e. rollouts whose group did not collapse.
- ``mixed_cuts/cost_multiplier``  rollouts per prompt / G. Both arms read 1.0 by design (D4).

Reward and length:
- ``reward/mean``, ``reward/var``, ``reward/acc``  over all rollouts (``acc`` = reward >=
  ``acc_threshold``); ``reward/within_group_var_mean``; ``reward/group_entropy`` (nats).
- ``reward/{std,cuts}/{mean,var,acc,n}`` and ``response_length/{std,cuts}/{mean,max}`` per kind.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from mixed_cuts.scheduler import CUTS, STD

ACR_TAU = 1e-6
"""tau in Definition 4.1 of He et al.: a group is collapsed when the population std of its rewards is below it."""

REWARD_ONE_EPS = 1e-6


def population_std(values: np.ndarray) -> float:
    """sigma_R of a group: standard deviation over the G rewards, dividing by G (ddof=0)."""
    return float(np.std(values, ddof=0)) if values.size else 0.0


def is_collapsed(values: np.ndarray, tau: float = ACR_TAU) -> bool:
    """Definition 4.1 indicator: I(sigma_R < tau)."""
    return values.size > 0 and population_std(values) < tau


def _entropy_nats(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(np.round(values, 6), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def variance_decomposition(r_std: np.ndarray, r_cuts: np.ndarray) -> dict[str, float]:
    """Eq. 5 terms for one group. Returns the five terms plus the measured mixed variance and residual."""
    mu_std, mu_cuts = float(r_std.mean()), float(r_cuts.mean())
    var_std, var_cuts = float(r_std.var(ddof=0)), float(r_cuts.var(ddof=0))
    between = 0.25 * (mu_std - mu_cuts) ** 2
    sigma2_mixed = float(np.concatenate([r_std, r_cuts]).var(ddof=0))
    rhs = 0.5 * (var_std + var_cuts) + between
    return {
        "mu_std": mu_std,
        "mu_cuts": mu_cuts,
        "var_std": var_std,
        "var_cuts": var_cuts,
        "between": between,
        "sigma2_mixed": sigma2_mixed,
        "identity_residual": abs(sigma2_mixed - rhs),
    }


def compute_group_diagnostics(
    uids: Sequence[Any],
    rewards: Sequence[float],
    rollout_kinds: Sequence[str] | None = None,
    response_lengths: Sequence[int] | None = None,
    *,
    acc_threshold: float = 0.5,
    group_size: int | None = None,
) -> dict[str, float]:
    """Compute the per-step diagnostics from flat per-rollout arrays.

    Args:
        uids: group id of every rollout (verl's ``uid``).
        rewards: scalar sequence-level reward of every rollout.
        rollout_kinds: ``"std"`` or ``"cuts"`` per rollout; ``None`` treats all as standard.
        response_lengths: response length in tokens per rollout (optional).
        acc_threshold: reward at or above which a rollout counts as correct.
        group_size: configured G, for ``cost_multiplier``; defaults to the mean observed group size.
    """
    uids_arr = np.asarray(list(uids), dtype=object)
    rewards_arr = np.asarray(list(rewards), dtype=np.float64)
    n = rewards_arr.size
    if uids_arr.size != n:
        raise ValueError(f"uids ({uids_arr.size}) and rewards ({n}) must have the same length")
    kinds_arr = (
        np.asarray([STD] * n, dtype=object)
        if rollout_kinds is None
        else np.asarray(list(rollout_kinds), dtype=object)
    )
    if kinds_arr.size != n:
        raise ValueError(f"rollout_kinds ({kinds_arr.size}) and rewards ({n}) must have the same length")
    lengths_arr = None if response_lengths is None else np.asarray(list(response_lengths), dtype=np.float64)
    if lengths_arr is not None and lengths_arr.size != n:
        raise ValueError(f"response_lengths ({lengths_arr.size}) and rewards ({n}) must have the same length")

    out: dict[str, float] = {}
    if n == 0:
        return out

    # ---- global reward statistics -----------------------------------------------------
    out["reward/mean"] = float(rewards_arr.mean())
    out["reward/var"] = float(rewards_arr.var(ddof=0))
    out["reward/acc"] = float((rewards_arr >= acc_threshold).mean())

    # ---- per-kind statistics -------------------------------------------------------------
    for kind in (STD, CUTS):
        mask = kinds_arr == kind
        out[f"reward/{kind}/n"] = float(mask.sum())
        if not mask.any():
            continue
        r = rewards_arr[mask]
        out[f"reward/{kind}/mean"] = float(r.mean())
        out[f"reward/{kind}/var"] = float(r.var(ddof=0))
        out[f"reward/{kind}/acc"] = float((r >= acc_threshold).mean())
        if lengths_arr is not None:
            ln = lengths_arr[mask]
            out[f"response_length/{kind}/mean"] = float(ln.mean())
            out[f"response_length/{kind}/max"] = float(ln.max())
    out["mixed_cuts/frac_cuts_rollouts"] = float((kinds_arr == CUTS).mean())

    # ---- group-level statistics ----------------------------------------------------------
    groups: dict[Any, list[int]] = {}
    for i, uid in enumerate(uids_arr):
        groups.setdefault(uid, []).append(i)

    n_groups = len(groups)
    n_collapsed = n_all_correct = n_all_wrong = 0
    n_std_present = n_std_collapsed = 0
    n_cuts_present = n_cuts_collapsed = 0
    n_both = n_saved = 0
    n_rollouts_in_collapsed = 0
    within_var: list[float] = []
    entropies: list[float] = []
    sizes: list[int] = []
    decomp_sums: dict[str, float] = {}
    for idx in groups.values():
        idx_arr = np.asarray(idx)
        r = rewards_arr[idx_arr]
        k = kinds_arr[idx_arr]
        sizes.append(r.size)
        collapsed = is_collapsed(r)
        n_collapsed += int(collapsed)
        if collapsed:
            n_rollouts_in_collapsed += r.size
            if abs(float(r[0]) - 1.0) <= REWARD_ONE_EPS:
                n_all_correct += 1
            elif abs(float(r[0])) <= REWARD_ONE_EPS:
                n_all_wrong += 1
        within_var.append(float(r.var(ddof=0)))
        entropies.append(_entropy_nats(r))

        r_std, r_cuts = r[k == STD], r[k == CUTS]
        std_collapsed = cuts_collapsed = None
        if r_std.size:
            n_std_present += 1
            std_collapsed = is_collapsed(r_std)
            n_std_collapsed += int(std_collapsed)
        if r_cuts.size:
            n_cuts_present += 1
            cuts_collapsed = is_collapsed(r_cuts)
            n_cuts_collapsed += int(cuts_collapsed)
        if r_std.size and r_cuts.size:
            n_both += 1
            if std_collapsed and not collapsed:
                n_saved += 1
            for key, value in variance_decomposition(r_std, r_cuts).items():
                decomp_sums[key] = decomp_sums.get(key, 0.0) + value

    out["mixed_cuts/n_groups"] = float(n_groups)
    out["mixed_cuts/group_size_mean"] = float(np.mean(sizes))
    out["mixed_cuts/advantage_collapse_rate"] = n_collapsed / n_groups
    out["mixed_cuts/all_correct_frac"] = n_all_correct / n_groups
    out["mixed_cuts/all_wrong_frac"] = n_all_wrong / n_groups
    out["reward/within_group_var_mean"] = float(np.mean(within_var))
    out["reward/group_entropy"] = float(np.mean(entropies))
    if n_std_present:
        out["mixed_cuts/std_subgroup_collapse_rate"] = n_std_collapsed / n_std_present
    if n_cuts_present:
        out["mixed_cuts/cuts_subgroup_collapse_rate"] = n_cuts_collapsed / n_cuts_present
    if n_both:
        out["mixed_cuts/frac_groups_saved_by_cuts"] = n_saved / n_both
        for key, total in decomp_sums.items():
            out[f"mixed_cuts/decomp/{key}"] = total / n_both

    # ---- cost accounting (He et al. Table 5) ----------------------------------------------
    out["mixed_cuts/rollouts_generated_per_step"] = float(n)
    out["mixed_cuts/sample_utilization"] = 1.0 - n_rollouts_in_collapsed / n
    g = float(group_size) if group_size else float(np.mean(sizes))
    out["mixed_cuts/cost_multiplier"] = (n / n_groups) / g if g else float("nan")
    return out


class RunningMean:
    """Running mean of a metric over the first ``max_steps`` steps (for ACR_100), resumable."""

    def __init__(self, max_steps: int = 100) -> None:
        self.max_steps = max_steps
        self._seen: dict[int, float] = {}

    def update(self, step: int, value: float) -> float | None:
        if 1 <= step <= self.max_steps and value == value:  # skip NaN
            self._seen[int(step)] = float(value)
        return self.value

    @property
    def value(self) -> float | None:
        return (sum(self._seen.values()) / len(self._seen)) if self._seen else None

    def rebuild(self, records: Sequence[dict[str, Any]], key: str) -> None:
        """Re-populate from ``metrics.jsonl`` records after a resume (uses ``record["step"]``)."""
        for rec in records:
            if key in rec and "step" in rec:
                self.update(int(rec["step"]), float(rec[key]))


# ------------------------------------------------------------------------------- D1 detector
def uniform_logprob_fraction(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    is_cuts: Sequence[bool],
    *,
    k: int,
    t_warm: int,
    k_min: int = 2,
    atol: float = 1e-4,
) -> tuple[float, int]:
    """Fraction of CUTS response tokens (after T_warm) whose old_log_prob equals -log(j), j in k_min..k.

    Under the correct D1 setup ``old_log_probs`` are recomputed by the actor (pi_theta_old) and this
    fraction is near chance. If the rollout engine's logprobs ever leaked into ``old_log_probs``,
    every CUTS token would carry exactly log(1/|S_t|) and the fraction would be ~1.0.

    ``k_min=2`` excludes j=1 (logprob 0) because near-deterministic tokens legitimately have
    logprob ~0 in healthy training; pass ``k_min=1`` for the informational variant.

    Returns (fraction, number_of_tokens_considered); fraction is 0.0 when nothing was considered.
    """
    lp = old_log_probs.detach().float()
    mask = response_mask.detach().bool().clone()
    if mask.shape != lp.shape:
        raise ValueError(f"old_log_probs {tuple(lp.shape)} and response_mask {tuple(mask.shape)} differ")
    rows = torch.as_tensor(list(is_cuts), dtype=torch.bool, device=lp.device)
    if rows.numel() != lp.shape[0]:
        raise ValueError(f"is_cuts has {rows.numel()} entries for {lp.shape[0]} rows")
    mask &= rows[:, None]
    if t_warm > 0:
        mask[:, : min(t_warm, mask.shape[1])] = False  # CUTS was inactive there
    n_tokens = int(mask.sum())
    if n_tokens == 0:
        return 0.0, 0
    targets = torch.tensor([-math.log(j) for j in range(max(1, k_min), k + 1)], device=lp.device)
    hits = (lp[mask][:, None] - targets[None, :]).abs().min(dim=1).values <= atol
    return float(hits.float().mean()), n_tokens
