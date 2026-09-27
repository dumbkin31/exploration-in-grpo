#!/usr/bin/env python
"""Fold a LoRA adapter into its base model and save a plain fp16 HuggingFace directory (for vLLM / eval).

verl's model merger writes a LoRA run as the base weights plus a separate ``lora_adapter/`` and does
NOT merge them (verl/model_merger/base_model_merger.py: save_lora_adapter), so evaluating that directory
would score the base model. ``scripts/merge_ckpt.sh`` calls this after the merger.

    python scripts/merge_lora.py <hf_dir_with_lora_adapter> [--out DIR] [--dtype float16]

Idempotent: a merged directory carries ``MERGED_LORA.json`` and is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "src", help="HF dir written by `python -m verl.model_merger merge` (base + lora_adapter/)"
    )
    ap.add_argument("--out", default=None, help="output dir (default: overwrite <src> in place)")
    ap.add_argument("--dtype", default="float16", help="dtype of the merged weights (sm_75: float16)")
    args = ap.parse_args()
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
