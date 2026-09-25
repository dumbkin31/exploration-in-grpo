from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mixed_cuts.diagnostics import (
    RunningMean,
    compute_group_diagnostics,
    is_collapsed,
    population_std,
    uniform_logprob_fraction,
    variance_decomposition,
)


def test_collapse_rate_and_saved_by_cuts():
    # group A: std all correct, cuts one wrong  -> std collapsed, whole group not -> saved
    # group B: everything correct                -> collapsed (all_correct)
    # group C: std mixed, cuts all wrong         -> nothing collapsed at group level, cuts subgroup collapsed
    uids = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    kinds = ["std", "std", "cuts", "cuts"] * 3
    rewards = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    lengths = [10, 20, 30, 40] * 3
    m = compute_group_diagnostics(uids, rewards, kinds, lengths, group_size=4)
    assert m["mixed_cuts/n_groups"] == 3 and m["mixed_cuts/group_size_mean"] == 4
    assert m["mixed_cuts/advantage_collapse_rate"] == pytest.approx(1 / 3)
    assert m["mixed_cuts/all_correct_frac"] == pytest.approx(1 / 3) and m["mixed_cuts/all_wrong_frac"] == 0.0
    assert m["mixed_cuts/std_subgroup_collapse_rate"] == pytest.approx(2 / 3)  # A and B
    assert m["mixed_cuts/cuts_subgroup_collapse_rate"] == pytest.approx(2 / 3)  # B and C
    assert m["mixed_cuts/frac_groups_saved_by_cuts"] == pytest.approx(1 / 3)  # only A
    assert m["mixed_cuts/frac_cuts_rollouts"] == 0.5
    assert m["reward/std/n"] == 6 and m["reward/cuts/n"] == 6
    assert m["reward/std/mean"] == pytest.approx(5 / 6) and m["reward/cuts/mean"] == pytest.approx(3 / 6)
    assert m["response_length/std/mean"] == 15 and m["response_length/cuts/mean"] == 35
    assert m["reward/mean"] == pytest.approx(8 / 12)
    # cost accounting: 12 rollouts, 4 (group B) contribute nothing
    assert m["mixed_cuts/rollouts_generated_per_step"] == 12
    assert m["mixed_cuts/sample_utilization"] == pytest.approx(8 / 12)
    assert m["mixed_cuts/cost_multiplier"] == 1.0


def test_d5_acr_uses_population_std_and_tau():
    """He et al. Def. 4.1: ACR = mean_j I(sigma_Rj < tau), sigma with ddof=0, tau = 1e-6."""
    assert population_std(np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(
        0.5
    )  # ddof=0 (ddof=1 would give 0.577)
    assert is_collapsed(np.array([1.0, 1.0, 1.0])) and is_collapsed(np.array([0.0, 0.0]))
    assert not is_collapsed(np.array([1.0, 1.0, 1.0 - 1e-3]))
    assert is_collapsed(np.array([0.5, 0.5 + 1e-9]))  # below tau -> collapsed
    uids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
    rewards = [1, 1, 1, 1] + [0, 0, 0, 0] + [1, 0, 1, 0] + [1, 1, 1, 0]
    m = compute_group_diagnostics(uids, rewards)
    assert m["mixed_cuts/advantage_collapse_rate"] == 0.5
    assert m["mixed_cuts/all_correct_frac"] == 0.25 and m["mixed_cuts/all_wrong_frac"] == 0.25
    assert m["mixed_cuts/sample_utilization"] == 0.5


def test_d6_variance_decomposition_identity_holds_for_equal_halves():
    """Eq. 5: sigma2_mixed = 0.5 (var_std + var_cuts) + 0.25 (mu_std - mu_cuts)^2, exact for 8/8."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        r_std = rng.integers(0, 2, size=8).astype(float)
        r_cuts = rng.integers(0, 2, size=8).astype(float)
        d = variance_decomposition(r_std, r_cuts)
        rhs = 0.5 * (d["var_std"] + d["var_cuts"]) + 0.25 * (d["mu_std"] - d["mu_cuts"]) ** 2
        assert d["sigma2_mixed"] == pytest.approx(rhs, abs=1e-12)
        assert d["identity_residual"] < 1e-9
        assert d["between"] == pytest.approx(0.25 * (r_std.mean() - r_cuts.mean()) ** 2)
    # a concrete case: std 8/8 correct, cuts 4/8 correct
    d = variance_decomposition(np.ones(8), np.array([1, 1, 1, 1, 0, 0, 0, 0.0]))
    assert d == pytest.approx(
        {
            "mu_std": 1.0,
            "mu_cuts": 0.5,
            "var_std": 0.0,
            "var_cuts": 0.25,
            "between": 0.0625,
            "sigma2_mixed": 0.1875,
            "identity_residual": 0.0,
        }
    )
    # the identity assumes equal halves: an unequal split leaves a residual
    d = variance_decomposition(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), np.array([0.0, 0.0]))
    assert d["identity_residual"] > 1e-3


def test_d6_terms_are_logged_per_step():
    uids = ["g1"] * 16 + ["g2"] * 16
    kinds = (["std"] * 8 + ["cuts"] * 8) * 2
    rewards = [1] * 8 + [1] * 4 + [0] * 4 + [1] * 8 + [0] * 8
    m = compute_group_diagnostics(uids, rewards, kinds, group_size=16)
    for key in ("mu_std", "mu_cuts", "var_std", "var_cuts", "between", "sigma2_mixed", "identity_residual"):
        assert f"mixed_cuts/decomp/{key}" in m
    assert m["mixed_cuts/decomp/mu_std"] == 1.0 and m["mixed_cuts/decomp/mu_cuts"] == 0.25
    assert m["mixed_cuts/decomp/between"] == pytest.approx((0.0625 + 0.25) / 2)
    assert m["mixed_cuts/decomp/identity_residual"] < 1e-9
    assert m["mixed_cuts/cost_multiplier"] == 1.0 and m["mixed_cuts/sample_utilization"] == 1.0


def test_group_entropy_and_within_group_variance():
    uids = ["g1"] * 4 + ["g2"] * 2
    rewards = [1, 1, 0, 0, 1, 1]
    m = compute_group_diagnostics(uids, rewards)
    assert m["reward/group_entropy"] == pytest.approx((math.log(2) + 0.0) / 2)
    assert m["reward/within_group_var_mean"] == pytest.approx((0.25 + 0.0) / 2)
    assert m["mixed_cuts/advantage_collapse_rate"] == 0.5
    assert m["reward/cuts/n"] == 0 and "reward/cuts/mean" not in m
    assert "mixed_cuts/frac_groups_saved_by_cuts" not in m and "mixed_cuts/decomp/mu_std" not in m
    assert m["mixed_cuts/std_subgroup_collapse_rate"] == 0.5


def test_all_collapsed_is_one_and_empty_is_empty():
    m = compute_group_diagnostics(["a", "a", "b", "b"], [1.0, 1.0, 0.0, 0.0])
    assert m["mixed_cuts/advantage_collapse_rate"] == 1.0 and m["mixed_cuts/sample_utilization"] == 0.0
    assert m["reward/group_entropy"] == 0.0
    assert compute_group_diagnostics([], []) == {}


def test_numpy_inputs_and_length_checks():
    m = compute_group_diagnostics(np.array(["a", "a"]), np.array([0.3, 0.9]), np.array(["std", "cuts"]))
    assert m["reward/acc"] == 0.5 and m["reward/cuts/acc"] == 1.0
    with pytest.raises(ValueError):
        compute_group_diagnostics(["a"], [1, 2])
    with pytest.raises(ValueError):
        compute_group_diagnostics(["a", "a"], [1, 2], ["std"])


def test_running_mean_acr_100_and_rebuild():
    rm = RunningMean(max_steps=100)
    assert rm.value is None
    rm.update(1, 0.5)
    rm.update(2, 0.7)
    rm.update(101, 100.0)  # beyond the window: ignored
    rm.update(3, float("nan"))
    assert rm.value == pytest.approx(0.6)
    rm2 = RunningMean(max_steps=100)
    rm2.rebuild(
        [{"step": 1, "k": 0.5}, {"step": 2, "k": 0.7}, {"step": 2, "k": 0.7}, {"step": 5, "other": 1}],
        key="k",
    )
    assert rm2.value == pytest.approx(0.6)  # a re-logged step after a resume counts once


def test_d1_detector_flags_engine_logprobs_but_not_recomputed_ones():
    k, t_warm = 5, 5
    T = 40
    # hazard: every CUTS token carries exactly -log|S_t| with |S_t| in 1..5
    sizes = torch.randint(1, k + 1, (4, T))
    hazard = -torch.log(sizes.float())
    mask = torch.ones(4, T)
    is_cuts = [True, True, False, False]
    frac_k2, n_tok = uniform_logprob_fraction(hazard, mask, is_cuts, k=k, t_warm=t_warm)
    assert n_tok == 2 * (T - t_warm)
    expected = float((sizes[:2, t_warm:] >= 2).float().mean())  # j=1 rows are excluded by k_min=2
    assert frac_k2 == pytest.approx(expected, abs=1e-6) and frac_k2 > 0.5
    frac_any, _ = uniform_logprob_fraction(hazard, mask, is_cuts, k=k, t_warm=t_warm, k_min=1)
    assert frac_any == pytest.approx(1.0)
    # healthy: actor-recomputed logprobs are continuous -> essentially never hit the lattice
    healthy = -torch.rand(4, T) * 3 - 0.01
    frac, _ = uniform_logprob_fraction(healthy, mask, is_cuts, k=k, t_warm=t_warm)
    assert frac < 0.05
    # standard rows and warm-up tokens are ignored; nothing considered -> (0, 0)
    assert uniform_logprob_fraction(hazard, mask, [False] * 4, k=k, t_warm=t_warm) == (0.0, 0)
    with pytest.raises(ValueError):
        uniform_logprob_fraction(hazard, mask[:, :10], is_cuts, k=k, t_warm=t_warm)
