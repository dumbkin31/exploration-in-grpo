#!/usr/bin/env python
"""Refuse to resume a run directory under a different training recipe (decision 018).

    python scripts/check_recipe.py RUN_DIR CONFIG [hydra overrides ...] [--verl-config-dir DIR]

``trainer.resume_mode: auto`` continues from whatever checkpoint RUN_DIR holds. When an earlier
submission of the same run trained under another recipe (a reused run name, a changed MC_MEMORY_PLAN, a
new MC_TOTAL_STEPS under a cosine schedule), the run would silently continue with mixed settings. This
composes the recipe of the coming submission the way the launcher does (CONFIG + the same overrides) and
compares it with the newest earlier job's saved config, ``RUN_DIR/jobs/*/hydra/.hydra/config.yaml``.

Exit 0: same recipe, or no checkpoint to resume. Exit 5: different recipe; MC_ALLOW_RECIPE_CHANGE=1
resumes anyway (the differences are still printed).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent

# What makes two submissions the same run: optimizer, loss, KL term, adapter, group split. The schedule's
# length matters only when the learning rate is not constant.
RECIPE_KEYS = (
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.actor.optim.lr_scheduler_type",
    "actor_rollout_ref.actor.loss_agg_mode",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.kl_loss_type",
    "actor_rollout_ref.actor.kl_loss_coef",
    "actor_rollout_ref.model.lora_rank",
    "actor_rollout_ref.model.lora_alpha",
    "mixed_cuts.n_std",
    "mixed_cuts.n_cuts",
)
SCHEDULE_LENGTH_KEY = "trainer.total_training_steps"


def recipe(cfg: Any) -> dict[str, Any]:
    out = {k: OmegaConf.select(cfg, k, default=None, throw_on_resolution_failure=False) for k in RECIPE_KEYS}
    if out["actor_rollout_ref.actor.optim.lr_scheduler_type"] not in (None, "constant"):
        out[SCHEDULE_LENGTH_KEY] = OmegaConf.select(
            cfg, SCHEDULE_LENGTH_KEY, default=None, throw_on_resolution_failure=False
        )
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
        except (TypeError, ValueError):
            return False
    return a == b


def previous_config(run_dir: Path) -> Path | None:
    """The newest earlier job's composed config, or None when the run has no saved job config."""
    configs = list(run_dir.glob("jobs/*/hydra/.hydra/config.yaml"))
    return max(configs, key=lambda p: p.stat().st_mtime) if configs else None


def differences(run_dir: Path, current: Any) -> tuple[Path | None, list[str]]:
    """(the earlier config compared with, one line per recipe key that differs)."""
    if not (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").exists():
        return None, []  # nothing to resume: the run starts from step 1
    prev_path = previous_config(run_dir)
    if prev_path is None:
        return None, []
    prev, cur = recipe(OmegaConf.load(prev_path)), recipe(current)
    lines = [
        f"  {k}: {prev.get(k)!r} (earlier job) -> {cur.get(k)!r} (this submission)"
        for k in sorted(set(prev) | set(cur))
        if not _same(prev.get(k), cur.get(k))
    ]
    return prev_path, lines


def _compose(config: str, overrides: list[str], verl_config_dir: str | None) -> Any:
    spec = importlib.util.spec_from_file_location("compose_config", REPO / "scripts" / "compose_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.compose_train_config(config, verl_config_dir, overrides)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="the Hydra overrides the launcher passes")
    ap.add_argument(
        "--verl-config-dir",
        default=os.environ.get("VERL_CONFIG_DIR") or os.environ.get("MC_VERL_CONFIG_DIR"),
        help="<verl checkout>/verl/trainer/config (default: the installed verl package)",
    )
    args = ap.parse_args(argv)
    current = _compose(args.config, args.overrides, args.verl_config_dir)
    prev_path, lines = differences(args.run_dir, current)
    if not lines:
        print(
            f"recipe check: OK ({'same recipe as ' + str(prev_path) if prev_path else 'nothing to resume'})"
        )
        return 0
    print(f"recipe check: {args.run_dir} holds a checkpoint from another recipe ({prev_path}):")
    print("\n".join(lines))
    if os.environ.get("MC_ALLOW_RECIPE_CHANGE") == "1":
        print("recipe check: MC_ALLOW_RECIPE_CHANGE=1, resuming anyway")
        return 0
    print(
        "Start the new recipe under a new run name (MC_RUN_TAG, MC_SEED or MC_RUN_NAME), or set "
        "MC_ALLOW_RECIPE_CHANGE=1 to resume this checkpoint with the new settings."
    )
    return 5


if __name__ == "__main__":
    sys.exit(main())
