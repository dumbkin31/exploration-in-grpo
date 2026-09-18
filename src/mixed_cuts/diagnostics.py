"""Group-level reward diagnostics for Mixed-CUTS (pure numpy, no verl).

Everything here answers *why* a run learns or does not, at the granularity GRPO cares about:
the prompt group. Inputs are flat per-rollout arrays (one entry per rollout) plus the group id
(``uid``) of each rollout and, optionally, its kind (``"std"`` / ``"cuts"``).

Metric glossary (see README section 7):

- ``mixed_cuts/advantage_collapse_rate``  fraction of groups whose rewards are all identical.
  For those groups the normalised advantage is exactly zero, so they contribute no gradient.
- ``mixed_cuts/std_subgroup_collapse_rate`` / ``cuts_subgroup_collapse_rate``  the same, for the
  standard / CUTS rollouts of each group taken alone (groups with that kind present).
- ``mixed_cuts/frac_groups_saved_by_cuts``  among groups that contain both kinds: the standard
  rollouts alone collapsed, but the mixed group did not. This is Mixed-CUTS's whole point.
- ``reward/mean``, ``reward/var``  over all rollouts; ``reward/within_group_var_mean`` is the mean
  over groups of the within-group variance (the raw GRPO signal); ``reward/group_entropy`` is the
  Shannon entropy (nats) of the reward histogram inside a group, averaged over groups.
- ``reward/{std,cuts}/{mean,var,acc,n}``  restricted to one kind; ``acc`` is the fraction of
  rollouts with reward >= ``acc_threshold`` (0.5 by default, i.e. "correct" for 0/1 rewards).
- ``response_length/{std,cuts}/{mean,max}``  trajectory length in tokens per kind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixed_cuts.scheduler import CUTS, STD

REWARD_EQ_EPS = 1e-6


def _is_collapsed(values: np.ndarray) -> bool:
    return values.size > 0 and float(values.max() - values.min()) <= REWARD_EQ_EPS


def _entropy_nats(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(np.round(values, 6), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def compute_group_diagnostics(
    uids: Sequence[Any],
    rewards: Sequence[float],
    rollout_kinds: Sequence[str] | None = None,
    response_lengths: Sequence[int] | None = None,
    *,
    acc_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the per-step diagnostics from flat per-rollout arrays.

    Args:
        uids: group id of every rollout (verl's ``uid``).
        rewards: scalar sequence-level reward of every rollout.
        rollout_kinds: ``"std"`` or ``"cuts"`` per rollout; ``None`` treats all as standard.
        response_lengths: response length in tokens per rollout (optional).
        acc_threshold: reward at or above which a rollout counts as correct.
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
    out["reward/var"] = float(rewards_arr.var())
    out["reward/acc"] = float((rewards_arr >= acc_threshold).mean())

    # ---- per-kind statistics -------------------------------------------------------------
    for kind in (STD, CUTS):
        mask = kinds_arr == kind
        out[f"reward/{kind}/n"] = float(mask.sum())
        if not mask.any():
            continue
        r = rewards_arr[mask]
        out[f"reward/{kind}/mean"] = float(r.mean())
        out[f"reward/{kind}/var"] = float(r.var())
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
    n_collapsed = 0
    n_std_present = n_std_collapsed = 0
    n_cuts_present = n_cuts_collapsed = 0
    n_both = n_saved = 0
    within_var = []
    entropies = []
    sizes = []
    for idx in groups.values():
        idx_arr = np.asarray(idx)
        r = rewards_arr[idx_arr]
        k = kinds_arr[idx_arr]
        sizes.append(r.size)
        collapsed = _is_collapsed(r)
        n_collapsed += int(collapsed)
        within_var.append(float(r.var()))
        entropies.append(_entropy_nats(r))

        r_std, r_cuts = r[k == STD], r[k == CUTS]
        std_collapsed = cuts_collapsed = None
        if r_std.size:
            n_std_present += 1
            std_collapsed = _is_collapsed(r_std)
            n_std_collapsed += int(std_collapsed)
        if r_cuts.size:
            n_cuts_present += 1
            cuts_collapsed = _is_collapsed(r_cuts)
            n_cuts_collapsed += int(cuts_collapsed)
        if r_std.size and r_cuts.size:
            n_both += 1
            if std_collapsed and not collapsed:
                n_saved += 1

    out["mixed_cuts/n_groups"] = float(n_groups)
    out["mixed_cuts/group_size_mean"] = float(np.mean(sizes))
    out["mixed_cuts/advantage_collapse_rate"] = n_collapsed / n_groups
    out["reward/within_group_var_mean"] = float(np.mean(within_var))
    out["reward/group_entropy"] = float(np.mean(entropies))
    if n_std_present:
        out["mixed_cuts/std_subgroup_collapse_rate"] = n_std_collapsed / n_std_present
    if n_cuts_present:
        out["mixed_cuts/cuts_subgroup_collapse_rate"] = n_cuts_collapsed / n_cuts_present
    if n_both:
        out["mixed_cuts/frac_groups_saved_by_cuts"] = n_saved / n_both
    return out
