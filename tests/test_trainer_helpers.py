"""Pure helpers of the trainer module (importable without verl? no: trainer imports verl).

The helpers are re-implemented in mixed_cuts.trainer; to test them without verl we import the
module lazily and skip when verl is absent, but the length helper is small enough that we also
keep a verl-free copy of the contract here via monkeypatching sys.modules.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch


@pytest.fixture
def trainer_module(monkeypatch):
    """Import mixed_cuts.trainer with verl/transfer_queue stubbed out (CPU dev env)."""
    if "mixed_cuts.trainer" in sys.modules:
        return sys.modules["mixed_cuts.trainer"]
    try:
        import verl  # noqa: F401

        import mixed_cuts.trainer as mod

        return mod
    except ImportError:
        pass
    stubs = {}
    tq = types.ModuleType("transfer_queue")
    stubs["transfer_queue"] = tq
    verl = types.ModuleType("verl")
    v1 = types.ModuleType("verl.trainer.ppo.v1")

    class PPOTrainerSync:  # minimal stand-in
        pass

    def register_trainer(name):
        return lambda cls: cls

    v1.PPOTrainerSync, v1.register_trainer = PPOTrainerSync, register_trainer
    dbg = types.ModuleType("verl.utils.debug")
    dbg.marked_timer = lambda *a, **k: __import__("contextlib").nullcontext()
    for name, m in {
        "verl": verl,
        "verl.trainer": types.ModuleType("verl.trainer"),
        "verl.trainer.ppo": types.ModuleType("verl.trainer.ppo"),
        "verl.trainer.ppo.v1": v1,
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.debug": dbg,
    }.items():
        stubs[name] = m
    for name, m in stubs.items():
        monkeypatch.setitem(sys.modules, name, m)
    import importlib

    mod = importlib.import_module("mixed_cuts.trainer")
    yield mod
    sys.modules.pop("mixed_cuts.trainer", None)


def test_padded_helper_and_id_lists(trainer_module):
    nested = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4])], layout=torch.jagged)
    assert trainer_module._padded(nested).tolist() == [[1, 2, 3], [4, 0, 0]]
    assert trainer_module._id_lists(nested) == [[1, 2, 3], [4]]
    assert trainer_module._id_lists(torch.tensor([[7, 8], [9, 0]])) == [[7, 8], [9, 0]]


def test_sequence_lengths_from_padded_mask(trainer_module):
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    assert trainer_module._sequence_lengths(mask) == [3, 1, 5]  # NOT [5, 5, 5]


def test_sequence_lengths_from_nested_mask(trainer_module):
    nested = torch.nested.nested_tensor([torch.ones(4), torch.ones(2)], layout=torch.jagged)
    assert trainer_module._sequence_lengths(nested) == [4, 2]
    assert trainer_module._sequence_lengths(None) is None
