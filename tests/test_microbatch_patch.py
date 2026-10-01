"""microbatch_patch: padded micro-batches hold sequences of similar length; outputs and gradient unchanged."""

from __future__ import annotations

import importlib
import inspect
import random
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from mixed_cuts import microbatch_patch as mp

# verl's call shape (verl/workers/engine/utils.py:112): the global name is looked up at call time
FAKE_UTILS = textwrap.dedent(
    """
    CALLS = []

    def rearrange_micro_batches(batch, max_token_len, dp_group=None, num_batches_divided_by=None,
                                same_micro_num_in_dp=True, min_num_micro_batch=None,
                                use_dynamic_bsz_balance=True, force_group_size=1):
        CALLS.append(max_token_len)
        return "verl", None

    def prepare_micro_batches(data, max_token_len):
        return rearrange_micro_batches(data, max_token_len=max_token_len, dp_group=None,
                                       same_micro_num_in_dp=True)
    """
)


def _lengths_like_step1(n: int = 2048, seed: int = 0) -> list[int]:
    """Prompt + response lengths matched to Mixed-CUTS step 1 (response mean 918, std 903, max 5000)."""
    rng = random.Random(seed)
    s2 = 0.68  # log(1 + (903 / 918) ** 2)
    out = []
    for _ in range(n):
        r = min(5000, max(82, int(rng.lognormvariate(6.822 - s2 / 2, s2**0.5))))
        out.append(max(44, min(805, int(rng.gauss(113, 45)))) + r)
    return out


@pytest.mark.parametrize("budget", [8192, 32768, 65536])
def test_partitions_cover_every_sequence_once_and_fit_the_budget_padded(budget):
    lengths = _lengths_like_step1()
    parts = mp.sorted_partitions(lengths, budget)
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(len(lengths)))
    for p in parts:
        assert len(p) * max(lengths[i] for i in p) <= budget
    longest = [max(lengths[i] for i in p) for p in parts]
    assert longest == sorted(longest, reverse=True), "longest micro-batch first (memory peak up front)"
    assert mp.padding_efficiency(lengths, parts) > 0.9
    if budget >= 32768:  # the configured budgets; near the longest length, packing is coarser
        # no more micro-batches than ~1/0.9 of verl's count, ceil(total / budget)
        assert len(parts) <= -(-sum(lengths) // budget) / 0.9 + 1


def test_a_sequence_longer_than_the_budget_stands_alone():
    assert mp.sorted_partitions([10, 50, 10], 40) == [[1], [0, 2]]


def test_steps_aside_where_its_assumptions_fail():
    base = {
        "dp_group": None,
        "force_group_size": 1,
        "num_batches_divided_by": None,
        "min_num_micro_batch": None,
    }
    assert mp.steps_aside(base, use_remove_padding=False) is None
    assert mp.steps_aside(base, use_remove_padding=True) == "remove-padding path"
    assert mp.steps_aside(dict(base, force_group_size=2), False) == "grouped samples"
    assert mp.steps_aside(dict(base, num_batches_divided_by=2), False) == "fixed micro-batch count"
    assert mp.steps_aside(dict(base, min_num_micro_batch=4), False) == "fixed micro-batch count"


@pytest.fixture
def fake_utils(tmp_path: Path, monkeypatch):
    pkg = f"mc_fake_engine_{uuid.uuid4().hex[:8]}"
    (tmp_path / pkg).mkdir()
    (tmp_path / pkg / "__init__.py").write_text("")
    (tmp_path / pkg / "utils.py").write_text(FAKE_UTILS)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("MC_SORTED_MICROBATCHES", raising=False)
    target = f"{pkg}.utils"
    yield target
    for name in [n for n in sys.modules if n.startswith(pkg)]:
        del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if getattr(f, "target", None) != target]


def test_installs_lazily_once_and_honours_the_opt_out(fake_utils, monkeypatch):
    mp.install(fake_utils)
    assert fake_utils not in sys.modules, "install must not import the target itself"
    mod = importlib.import_module(fake_utils)
    wrapped = mod.rearrange_micro_batches
    assert getattr(wrapped, mp._FLAG) and wrapped.__wrapped__.__name__ == "rearrange_micro_batches"
    mp.install(fake_utils)
    assert mod.rearrange_micro_batches is wrapped, "idempotent"
    assert not any(getattr(f, "target", None) == fake_utils for f in sys.meta_path), "one-shot finder"


def test_opt_out_leaves_verl_alone(fake_utils, monkeypatch):
    monkeypatch.setenv("MC_SORTED_MICROBATCHES", "0")
    mp.install(fake_utils)
    mod = importlib.import_module(fake_utils)
    assert not hasattr(mod.rearrange_micro_batches, mp._FLAG)


def _nested_batch(lengths, use_remove_padding):
    torch = pytest.importorskip("torch")
    tu = pytest.importorskip("verl.utils.tensordict_utils")
    from tensordict import TensorDict

    ids = [torch.arange(n) + 1000 * k for k, n in enumerate(lengths)]
    batch = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(ids, layout=torch.jagged),
            "row": torch.arange(len(lengths)),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(batch, use_remove_padding=use_remove_padding)
    return batch


def test_real_verl_partition_restores_order_and_saves_padding():
    """With verl installed: the patched call returns micro-batches verl's restore puts back in order."""
    sb = pytest.importorskip("verl.utils.seqlen_balancing")
    utils = pytest.importorskip("verl.workers.engine.utils")
    lengths = _lengths_like_step1(n=512, seed=1)  # one update mini-batch: 32 prompts x 16
    budget = 32768
    batch = _nested_batch(lengths, use_remove_padding=False)
    orig = utils.rearrange_micro_batches
    try:
        assert mp.apply(utils)
        mbs, idx = utils.rearrange_micro_batches(batch, max_token_len=budget, dp_group=None)
        assert [mb["input_ids"].offsets().diff().tolist() for mb in mbs] == [
            [lengths[i] for i in p] for p in idx
        ]
        rows = sb.restore_dynamic_batch(sb.torch.cat([mb["row"] for mb in mbs]), idx)
        assert rows.tolist() == list(range(len(lengths)))
        ours = mp.padding_efficiency(lengths, idx)
        _, verl_idx = orig(batch, max_token_len=budget, dp_group=None)
        assert ours > 0.9 and ours > 2 * mp.padding_efficiency(lengths, verl_idx)
        # remove-padding batches go to verl untouched
        rm = _nested_batch(lengths, use_remove_padding=True)
        assert utils.rearrange_micro_batches(rm, max_token_len=budget)[1] == orig(rm, max_token_len=budget)[1]
    finally:
        utils.rearrange_micro_batches = orig


def test_verl_still_splits_where_the_patch_expects():
    """Guard for verl upgrades: the call site, the signature and the mini-batch loss normalisation."""
    utils = pytest.importorskip("verl.workers.engine.utils")
    impl = pytest.importorskip("verl.workers.engine.fsdp.transformer_impl")
    params = list(inspect.signature(inspect.unwrap(utils.rearrange_micro_batches)).parameters)
    assert params[:2] == ["batch", "max_token_len"]
    assert {"dp_group", "num_batches_divided_by", "min_num_micro_batch", "force_group_size"} <= set(params)
    assert "rearrange_micro_batches(" in inspect.getsource(utils.prepare_micro_batches)
    fb = inspect.getsource(impl.FSDPEngine.forward_backward_batch)
    assert fb.index('data["loss_mask"].sum()') < fb.index("prepare_micro_batches("), (
        "the loss token count must come from the whole mini-batch, before the split"
    )
    assert "restore_dynamic_batch" in inspect.getsource(utils.postprocess_batch_func)
