#!/usr/bin/env python
"""Fold a LoRA adapter into its base model and save a plain fp16 HuggingFace directory (for vLLM / eval).

verl's model merger writes a LoRA run as the base weights plus a separate ``lora_adapter/`` and does
NOT merge them (verl/model_merger/base_model_merger.py: save_lora_adapter), so evaluating that directory
would score the base model. ``scripts/merge_ckpt.sh`` calls this after the merger.

    python scripts/merge_lora.py <hf_dir_with_lora_adapter> [--out DIR] [--dtype float16]
    python scripts/merge_lora.py --verl-actor-dir <ckpt>/global_step_N/actor --base <model dir> --out DIR

The second form handles ``checkpoint.save_lora_only`` checkpoints (the default since decision 012):
``model_world_size_1_rank_0.pt`` then holds only the ``lora_``/``.adapter_`` keys, verl's merger would
assert on the missing base keys, so the PEFT adapter is rebuilt from it the way the merger does
(base_model_merger.py: save_lora_adapter; r/alpha from ``lora_train_meta.json``) and merged into the
original base model. Idempotent: a merged directory carries ``MERGED_LORA.json`` and is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def is_lora_only_state_dict(state: dict) -> bool:
    """verl's own test (fsdp_checkpoint_manager.is_lora_only_state_dict): every key is an adapter key."""
    return bool(state) and all(("lora_" in k or ".adapter_" in k) for k in state)


def adapter_from_verl_lora_only(actor_dir: Path, adapter_dir: Path) -> Path:
    """Rebuild a PEFT adapter directory from verl's LoRA-only FSDP checkpoint (world size 1).

    Mirrors verl.model_merger.base_model_merger.save_lora_adapter: strip PEFT's ``.default`` adapter name
    from the keys, infer target modules from the key names, take r / lora_alpha / task_type from
    lora_train_meta.json (refuse to guess alpha: the scaling would be silently wrong).
    """
    import torch
    from safetensors.torch import save_file

    pts = sorted(actor_dir.glob("model_world_size_*_rank_0.pt"))
    if not pts:
        raise SystemExit(f"no model_world_size_*_rank_0.pt under {actor_dir}")
    if len(list(actor_dir.glob("model_world_size_*_rank_*.pt"))) > 1:
        raise SystemExit("more than one rank shard: run `python -m verl.model_merger merge` (FSDP world > 1)")
    state = torch.load(pts[0], map_location="cpu", weights_only=False)
    if not is_lora_only_state_dict(state):
        raise SystemExit(
            f"{pts[0]} is not a save_lora_only checkpoint: use the verl merger + plain merge path"
        )
    lora: dict = {}
    targets: set[str] = set()
    for name, value in state.items():
        if "lora_" not in name:
            continue  # `.adapter_` keys (modules_to_save) are not used by our config
        key = name.replace(".default.weight", ".weight")
        targets.add(key.split(".")[-3])
        tensor = value.full_tensor() if hasattr(value, "full_tensor") else value
        lora[key] = tensor.detach().to("cpu").contiguous()
    meta_path = actor_dir / "lora_train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sample = next(iter(lora.values()))
    cfg = {
        "peft_type": "LORA",
        "r": int(meta.get("r") or min(sample.shape)),
        "lora_alpha": int(meta.get("lora_alpha") or 0),
        "target_modules": sorted(targets),
        "task_type": meta.get("task_type") or "CAUSAL_LM",
        "bias": "none",
        "inference_mode": True,
    }
    if cfg["lora_alpha"] == 0:
        raise SystemExit(f"lora_alpha missing in {meta_path}; refusing to guess")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    save_file(lora, str(adapter_dir / "adapter_model.safetensors"))
    print(
        f"rebuilt PEFT adapter from {pts[0].name}: {len(lora)} tensors, r={cfg['r']}, "
        f"alpha={cfg['lora_alpha']}, targets={cfg['target_modules']}"
    )
    return adapter_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "src", nargs="?", help="HF dir written by `python -m verl.model_merger merge` (base + lora_adapter/)"
    )
    ap.add_argument(
        "--verl-actor-dir", default=None, help="LoRA-only verl checkpoint dir (global_step_N/actor)"
    )
    ap.add_argument("--base", default=None, help="base model dir (with --verl-actor-dir)")
    ap.add_argument("--out", default=None, help="output dir (default: overwrite <src> in place)")
    ap.add_argument("--dtype", default="float16", help="dtype of the merged weights (sm_75: float16)")
    args = ap.parse_args()
    if args.verl_actor_dir:
        if not (args.base and args.out):
            ap.error("--verl-actor-dir needs --base and --out")
        src = Path(args.base)
        out = Path(args.out)
        marker = out / "MERGED_LORA.json"
        if marker.exists():
            print(f"already merged: {marker.read_text().strip()}")
            return 0
        adapter = adapter_from_verl_lora_only(Path(args.verl_actor_dir), out / "lora_adapter")
    else:
        if not args.src:
            ap.error("give <src> or --verl-actor-dir")
        src = Path(args.src)
        out = Path(args.out) if args.out else src
        adapter = src / "lora_adapter"
        marker = out / "MERGED_LORA.json"
        if marker.exists():
            print(f"already merged: {marker.read_text().strip()}")
            return 0
        if not adapter.is_dir():
            print(f"no lora_adapter/ under {src}: a full fine-tune checkpoint, nothing to merge")
            return 0

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    print(f"loading base from {src} in {args.dtype}")
    base = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, device_map="cpu")
    print(f"applying adapter {adapter}")
    model = PeftModel.from_pretrained(base, adapter)
    merged = model.merge_and_unload()
    n_params = sum(p.numel() for p in merged.parameters())
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    marker.write_text(
        json.dumps(
            {
                "merged_from": str(adapter),
                "r": cfg.get("r"),
                "lora_alpha": cfg.get("lora_alpha"),
                "target_modules": cfg.get("target_modules"),
                "dtype": args.dtype,
                "params": n_params,
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    print(f"merged {n_params / 1e9:.2f} B params -> {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
