from __future__ import annotations

import math

import numpy as np
import pytest

from mixed_cuts.diagnostics import compute_group_diagnostics


def test_collapse_rate_and_saved_by_cuts():
    # group A: std all correct, cuts one wrong  -> std collapsed, whole group not -> saved
    # group B: everything correct                -> collapsed everywhere
    # group C: std mixed, cuts all wrong         -> nothing collapsed at group level, cuts subgroup collapsed
    uids = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    kinds = ["std", "std", "cuts", "cuts"] * 3
    rewards = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    lengths = [10, 20, 30, 40] * 3
    m = compute_group_diagnostics(uids, rewards, kinds, lengths)
    assert m["mixed_cuts/n_groups"] == 3 and m["mixed_cuts/group_size_mean"] == 4
    assert m["mixed_cuts/advantage_collapse_rate"] == pytest.approx(1 / 3)
    assert m["mixed_cuts/std_subgroup_collapse_rate"] == pytest.approx(2 / 3)  # A and B
    assert m["mixed_cuts/cuts_subgroup_collapse_rate"] == pytest.approx(2 / 3)  # B and C
    assert m["mixed_cuts/frac_groups_saved_by_cuts"] == pytest.approx(1 / 3)  # only A
    assert m["mixed_cuts/frac_cuts_rollouts"] == 0.5
    assert m["reward/std/n"] == 6 and m["reward/cuts/n"] == 6
    assert m["reward/std/mean"] == pytest.approx(5 / 6) and m["reward/cuts/mean"] == pytest.approx(3 / 6)
    assert m["reward/std/acc"] == pytest.approx(5 / 6)
    assert m["response_length/std/mean"] == 15 and m["response_length/cuts/mean"] == 35
    assert m["response_length/cuts/max"] == 40
    assert m["reward/mean"] == pytest.approx(8 / 12)


def test_group_entropy_and_within_group_variance():
    uids = ["g1"] * 4 + ["g2"] * 2
    rewards = [1, 1, 0, 0, 1, 1]
    m = compute_group_diagnostics(uids, rewards)
    assert m["reward/group_entropy"] == pytest.approx((math.log(2) + 0.0) / 2)
    assert m["reward/within_group_var_mean"] == pytest.approx((0.25 + 0.0) / 2)
    assert m["mixed_cuts/advantage_collapse_rate"] == 0.5
    # no cuts rollouts at all: no cuts keys, no "saved" key
    assert m["reward/cuts/n"] == 0 and "reward/cuts/mean" not in m
    assert "mixed_cuts/frac_groups_saved_by_cuts" not in m
    assert m["mixed_cuts/std_subgroup_collapse_rate"] == 0.5


def test_all_collapsed_is_one_and_empty_is_empty():
    m = compute_group_diagnostics(["a", "a", "b", "b"], [1.0, 1.0, 0.0, 0.0])
    assert m["mixed_cuts/advantage_collapse_rate"] == 1.0
    assert m["reward/group_entropy"] == 0.0
    assert compute_group_diagnostics([], []) == {}


def test_numpy_inputs_and_length_checks():
    m = compute_group_diagnostics(np.array(["a", "a"]), np.array([0.3, 0.9]), np.array(["std", "cuts"]))
    assert m["reward/acc"] == 0.5 and m["reward/cuts/acc"] == 1.0
    with pytest.raises(ValueError):
        compute_group_diagnostics(["a"], [1, 2])
    with pytest.raises(ValueError):
        compute_group_diagnostics(["a", "a"], [1, 2], ["std"])
