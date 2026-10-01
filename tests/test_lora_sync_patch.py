"""lora_sync_patch: verl's merged-LoRA weight sync runs without the 6.9 GB CPU backup of the base model."""

from __future__ import annotations

import importlib
import inspect
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from mixed_cuts import lora_sync_patch, worker_hooks

FAKE_ENGINE = textwrap.dedent(
    """
    import contextlib

    CALLS = []

    @contextlib.contextmanager
    def merged_lora_context(actor, backup_adapters=False):
        CALLS.append(backup_adapters)
        yield

    def sync(actor):
        # the shape of FSDPEngine._merged_lora_per_tensor_param: the global name is looked up at call time
        with merged_lora_context(actor, backup_adapters=True):
            return "synced"
    """
)


@pytest.fixture
def fake_engine(tmp_path: Path, monkeypatch):
    pkg = f"mc_fake_verl_{uuid.uuid4().hex[:8]}"
    (tmp_path / pkg).mkdir()
    (tmp_path / pkg / "__init__.py").write_text("")
    (tmp_path / pkg / "transformer_impl.py").write_text(FAKE_ENGINE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("MC_LORA_SYNC_BACKUP", raising=False)
    target = f"{pkg}.transformer_impl"
    yield target
    for name in [n for n in sys.modules if n.startswith(pkg)]:
        del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if getattr(f, "target", None) != target]


class _Actor:
    def modules(self):
        return iter(())


def test_patch_applies_when_the_engine_is_imported_later(fake_engine):
    lora_sync_patch.install(fake_engine)
    assert fake_engine not in sys.modules, "install must not import the engine itself"
    mod = importlib.import_module(fake_engine)
    assert mod.sync(_Actor()) == "synced"
    assert mod.CALLS == [False], "the engine asked for backup_adapters=True; the patch must pass False"
    assert not any(getattr(f, "target", None) == fake_engine for f in sys.meta_path), "one-shot finder"


def test_patch_applies_to_an_already_imported_engine_and_is_idempotent(fake_engine):
    mod = importlib.import_module(fake_engine)
    lora_sync_patch.install(fake_engine)
    wrapped = mod.merged_lora_context
    lora_sync_patch.install(fake_engine)
    assert mod.merged_lora_context is wrapped
    mod.sync(_Actor())
    assert mod.CALLS == [False]


def test_opt_out_keeps_verls_backup(fake_engine, monkeypatch):
    monkeypatch.setenv("MC_LORA_SYNC_BACKUP", "1")
    lora_sync_patch.install(fake_engine)
    mod = importlib.import_module(fake_engine)
    mod.sync(_Actor())
    assert mod.CALLS == [True]


def test_a_layer_left_merged_after_the_sync_is_an_error(fake_engine, monkeypatch):
    mod = importlib.import_module(fake_engine)
    lora_sync_patch.install(fake_engine)
    monkeypatch.setattr(lora_sync_patch, "_any_merged_lora", lambda actor: True)
    with pytest.raises(RuntimeError, match="still merged"):
        mod.sync(_Actor())


def test_worker_hook_installs_both_patches(monkeypatch):
    called = []
    monkeypatch.setattr("mixed_cuts.sdpa_patch.install", lambda: called.append("sdpa"))
    monkeypatch.setattr("mixed_cuts.lora_sync_patch.install", lambda: called.append("lora_sync"))
    monkeypatch.setattr("mixed_cuts.zmq_socket_patch.install", lambda: called.append("zmq"))
    monkeypatch.setattr("mixed_cuts.microbatch_patch.install", lambda: called.append("microbatch"))
    worker_hooks.install()
    assert called == ["sdpa", "lora_sync", "zmq", "microbatch"]


def test_merge_unmerge_round_trip_drift_is_far_below_fp16():
    """100 sync cycles of W += s*B@A; W -= s*B@A in fp32, with fresh adapters each step (what PEFT does)."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(256, 512, generator=g) * 0.02
    w = w0.clone()
    for _ in range(100):
        a = torch.randn(64, 512, generator=g) * 0.01
        b = torch.randn(256, 64, generator=g) * 0.01
        delta = (b @ a) * 2.0  # lora_alpha / r = 128 / 64
        w += delta
        w -= delta
    rel = float((w - w0).abs().max() / w0.abs().max())
    assert rel < 1e-6, rel  # fp16 resolution is ~1e-3 relative


def test_real_peft_unmerge_restores_the_base(monkeypatch):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    base = torch.nn.Sequential(torch.nn.Linear(64, 64))
    w0 = base[0].weight.detach().clone()
    model = peft.get_peft_model(base, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["0"]))
    layer = next(m for m in model.modules() if isinstance(m, peft.tuners.lora.LoraLayer))
    torch.nn.init.normal_(layer.lora_B["default"].weight, std=0.02)
    layer.merge()
    assert not torch.equal(layer.get_base_layer().weight, w0)
    layer.unmerge()
    assert not layer.merged
    assert torch.allclose(layer.get_base_layer().weight, w0, rtol=0, atol=1e-7)


def test_verl_still_hardcodes_the_backup_in_the_sync_path():
    """Guard for verl upgrades: if this changes, re-check whether the patch is still needed or still valid."""
    impl = pytest.importorskip("verl.workers.engine.fsdp.transformer_impl")
    assert hasattr(impl, "merged_lora_context")
    src = inspect.getsource(impl.FSDPEngine._merged_lora_per_tensor_param)
    assert "merged_lora_context(self.module, backup_adapters=True)" in src
    fsdp_utils = importlib.import_module("verl.utils.fsdp_utils")
    ctx_src = inspect.getsource(fsdp_utils.merged_lora_context)
    assert "fsdp_merge_unmerge(actor, do_merge=False)" in ctx_src, "the no-backup fallback must unmerge"
