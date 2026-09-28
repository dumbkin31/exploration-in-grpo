"""merge_lora.py rebuilds a PEFT adapter from verl's LoRA-only FSDP checkpoint (checkpoint.save_lora_only)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("merge_lora", REPO / "scripts" / "merge_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _fake_lora_only_ckpt(actor: Path, *, meta: bool = True) -> dict:
    prefix = "base_model.model.model.layers.0."
    state = {
        prefix + "self_attn.q_proj.lora_A.default.weight": torch.zeros(4, 16),
        prefix + "self_attn.q_proj.lora_B.default.weight": torch.ones(16, 4),
        prefix + "mlp.down_proj.lora_A.default.weight": torch.zeros(4, 8),
        prefix + "mlp.down_proj.lora_B.default.weight": torch.ones(8, 4),
    }
    actor.mkdir(parents=True)
    torch.save(state, actor / "model_world_size_1_rank_0.pt")
    if meta:
        (actor / "lora_train_meta.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 8, "task_type": "CAUSAL_LM"})
        )
    return state


def test_adapter_rebuilt_like_verls_merger(tmp_path: Path):
    from safetensors.torch import load_file

    mod = _load()
    actor = tmp_path / "global_step_3" / "actor"
    _fake_lora_only_ckpt(actor)
    out = mod.adapter_from_verl_lora_only(actor, tmp_path / "hf" / "lora_adapter")
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["r"] == 4 and cfg["lora_alpha"] == 8 and cfg["peft_type"] == "LORA"
    assert cfg["target_modules"] == ["down_proj", "q_proj"]
    tensors = load_file(str(out / "adapter_model.safetensors"))
    # PEFT's `.default` adapter name is stripped, as in verl.model_merger.base_model_merger.save_lora_adapter
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in tensors
    assert all(".default." not in k for k in tensors)
    assert tensors["base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"].shape == (8, 4)


def test_refuses_to_guess_alpha_without_meta(tmp_path: Path):
    mod = _load()
    actor = tmp_path / "actor"
    _fake_lora_only_ckpt(actor, meta=False)
    with pytest.raises(SystemExit, match="lora_alpha"):
        mod.adapter_from_verl_lora_only(actor, tmp_path / "lora_adapter")


def test_full_state_dict_is_not_lora_only(tmp_path: Path):
    mod = _load()
    actor = tmp_path / "actor"
    actor.mkdir()
    torch.save(
        {"base_model.model.lm_head.weight": torch.zeros(2, 2), "x.lora_A.default.weight": torch.zeros(2, 2)},
        actor / "model_world_size_1_rank_0.pt",
    )
    assert not mod.is_lora_only_state_dict({"a.weight": 1, "b.lora_A.weight": 2})
    with pytest.raises(SystemExit, match="not a save_lora_only"):
        mod.adapter_from_verl_lora_only(actor, tmp_path / "lora_adapter")
