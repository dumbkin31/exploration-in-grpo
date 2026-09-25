#!/usr/bin/env python
"""Evaluate a checkpoint on MATH-500 / AIME24 / AIME25 / AMC23 / GPQA-diamond.

    python eval/run_eval.py --ckpt /share1/$USER/mixed-cuts/runs/$USER/<job>/hf/global_step_50 \\
        --benchmarks math500,aime24 --n-samples 16 [--config configs/eval/default.yaml] [--out DIR]

``--ckpt`` must be a HuggingFace model directory. For a verl FSDP checkpoint
(``.../global_step_N/actor``) run ``scripts/merge_ckpt.sh`` first.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mc_eval.runner import run_all  # noqa: E402


def load_cfg(path: str) -> dict:
    """Load an eval YAML; supports the tiny `defaults: [default, _self_]` layering used in configs/eval."""
    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", None)
    base = OmegaConf.create()
    if defaults:
        for d in defaults:
            if d == "_self_":
                continue
            base = OmegaConf.merge(base, load_cfg(str(Path(path).parent / f"{d}.yaml")))
    return OmegaConf.to_container(OmegaConf.merge(base, cfg), resolve=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="HF model directory (or a HF Hub id for the base model)")
    ap.add_argument("--benchmarks", default=None, help="comma-separated; default from config")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--config", default=str(REPO / "configs" / "eval" / "default.yaml"))
    ap.add_argument("--data-dir", default=os.environ.get("MC_DATA_DIR"))
    ap.add_argument("--out", default=None, help="default: $MC_RUNS_DIR/eval/<ckpt-name>-<timestamp>")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems per benchmark (debug)")
    ap.add_argument("--tp", type=int, default=None, help="tensor parallel size override")
    ap.add_argument(
        "--seed", type=int, default=None, help="sampling + engine + bootstrap seed (default from config)"
    )
    ap.add_argument(
        "--override", action="append", default=[], help="dotlist override, e.g. sampling.temperature=1.0"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_cfg(args.config)
    if args.override:
        cfg = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(args.override)), resolve=True
        )
    if args.n_samples:
        cfg["n_samples"] = args.n_samples
    if args.tp:
        cfg["engine"]["tensor_parallel_size"] = args.tp
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))
    cfg["sampling"]["seed"] = seed
    cfg["engine"]["seed"] = seed
    cfg["bootstrap_seed"] = seed
    benchmarks = args.benchmarks.split(",") if args.benchmarks else list(cfg["benchmarks"])
    limit = args.limit or cfg.get("limit")
    if not args.data_dir:
        print("ERROR: --data-dir or MC_DATA_DIR required (source configs/ada.env.sh)", file=sys.stderr)
        return 2

    ckpt = Path(args.ckpt)
    if ckpt.is_dir() and not (ckpt / "config.json").exists():
        hint = "scripts/merge_ckpt.sh <run_dir> [global_step]"
        print(
            f"ERROR: {ckpt} is not a HF model dir (no config.json). If it is a verl checkpoint, run: {hint}",
            file=sys.stderr,
        )
        return 2

    out = args.out or os.path.join(
        os.environ.get("MC_RUNS_DIR", str(REPO / "runs")),
        "eval",
        f"{ckpt.name}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    from mc_eval.generate import VllmGenerator

    generator = VllmGenerator(str(ckpt), cfg["engine"], cfg.get("chat_template_kwargs"))
    run_all(benchmarks, args.data_dir, generator, cfg, out, limit=limit)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
