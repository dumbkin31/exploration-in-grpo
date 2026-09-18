#!/usr/bin/env python
"""Compose a training config exactly as `mixed_cuts.main` would, and print or check it.

Catches YAML typos, broken interpolations and wrong key paths *before* burning a GPU
allocation. Works without verl installed by pointing ``--verl-config-dir`` at a verl checkout
(``verl/trainer/config``); on the cluster it defaults to the installed package.

    python scripts/compose_config.py smoke                      # dump the composed YAML
    python scripts/compose_config.py smoke --check              # assert the invariants below
    python scripts/compose_config.py math_mixed_cuts --verl-config-dir /path/to/verl/trainer/config

Invariants checked with --check (each one is a lesson from section 1/5 of the README):
  * no 'bfloat16'/'bf16' anywhere under actor_rollout_ref (sm_75 has no bf16)
  * rollout.n == mixed_cuts.n_std + n_cuts when mixed_cuts.enabled
  * the CUTS logits processor is registered and the mixed agent-loop manager is selected
  * attn_implementation == sdpa and use_remove_padding == false (no FlashAttention-2)
  * trainer.use_v1 and trainer_mode == mixed_cuts_sync
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
TRAIN_CONFIG_DIR = REPO / "configs" / "train"


def compose_train_config(name: str, verl_config_dir: str | None, overrides: list[str]):
    searchpath = [f"file://{verl_config_dir}"] if verl_config_dir else ["pkg://verl.trainer.config"]
    fake_paths = [
        "paths.run_dir=/tmp/mc-compose/run",
        "paths.model_dir=/tmp/mc-compose/model",
        "paths.data_dir=/tmp/mc-compose/data",
        "paths.repo_dir=" + str(REPO),
    ]
    with initialize_config_dir(config_dir=str(TRAIN_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=name,
            overrides=[f"hydra.searchpath=[{','.join(searchpath)}]", *fake_paths, *overrides],
        )
    return cfg


def _walk(node, path=""):
    if OmegaConf.is_dict(node):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif OmegaConf.is_list(node):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def check(cfg) -> list[str]:
    problems = []
    arr = cfg.actor_rollout_ref
    for path, value in _walk(arr, "actor_rollout_ref"):
        if isinstance(value, str) and value.lower() in ("bf16", "bfloat16"):
            problems.append(f"{path} = {value} (bf16 is unsupported on sm_75)")
    mc = cfg.get("mixed_cuts")
    if mc and mc.enabled and (mc.n_std + mc.n_cuts) != arr.rollout.n:
        problems.append(f"mixed_cuts.n_std + n_cuts = {mc.n_std + mc.n_cuts} != rollout.n = {arr.rollout.n}")
    lps = list(arr.rollout.engine_kwargs.vllm.get("logits_processors", []) or [])
    if mc and mc.enabled and mc.n_cuts > 0 and "cuts.vllm_logits_processor:CutsLogitsProcessor" not in lps:
        problems.append(f"CUTS enabled but logits processor not registered: {lps}")
    mgr = arr.rollout.agent.get("agent_loop_manager_class")
    if mgr != "mixed_cuts.agent_loop.MixedCutsAgentLoopManagerTQ":
        problems.append(f"rollout.agent.agent_loop_manager_class = {mgr}")
    if arr.model.get("override_config", {}).get("attn_implementation") != "sdpa":
        problems.append(
            "model.override_config.attn_implementation must be sdpa (no FlashAttention-2 on sm_75)"
        )
    if arr.model.use_remove_padding:
        problems.append("model.use_remove_padding must be false (needs FA2 varlen kernels)")
    if arr.rollout.dtype != "float16":
        problems.append(f"rollout.dtype = {arr.rollout.dtype}")
    if not cfg.trainer.use_v1 or cfg.trainer.v1.trainer_mode != "mixed_cuts_sync":
        problems.append(f"trainer.use_v1={cfg.trainer.use_v1} trainer_mode={cfg.trainer.v1.trainer_mode}")
    if (
        cfg.reward.custom_reward_function.get("path") in (None, "")
        or cfg.reward.custom_reward_function.name != "compute_score"
    ):
        problems.append("reward.custom_reward_function must point at mixed_cuts.reward:compute_score")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="name under configs/train (without .yaml)")
    ap.add_argument(
        "--verl-config-dir", default=None, help="verl/trainer/config of a checkout (default: installed verl)"
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--select", default=None, help="only print this subtree, e.g. actor_rollout_ref.rollout")
    ap.add_argument("overrides", nargs="*", help="extra hydra overrides")
    args = ap.parse_args()
    cfg = compose_train_config(args.config, args.verl_config_dir, args.overrides)
    if args.check:
        problems = check(cfg)
        for p in problems:
            print("FAIL ", p)
        print(f"== {args.config}: {'OK' if not problems else str(len(problems)) + ' problem(s)'} ==")
        return 1 if problems else 0
    node = OmegaConf.select(cfg, args.select) if args.select else cfg
    print(OmegaConf.to_yaml(node, resolve=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
