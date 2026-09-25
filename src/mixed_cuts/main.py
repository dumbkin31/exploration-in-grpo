"""Training entry point: verl's ``main_ppo`` with the Mixed-CUTS pieces registered.

This mirrors ``verl/trainer/main_ppo.py`` (v0.9.0). Differences:

1. the Hydra primary config is ours (``configs/train/<name>.yaml``), layered on verl's
   ``ppo_trainer`` through ``hydra.searchpath=[pkg://verl.trainer.config]``;
2. the Ray task runner imports :mod:`mixed_cuts.trainer` (which registers the
   ``mixed_cuts_sync`` trainer) *inside the actor process* before verl looks the trainer up;
3. the top-level ``seed`` seeds ``random``, ``numpy`` and ``torch`` in the driver (verl's own
   seeds are wired from the same value in the YAML);
4. a :class:`MixedCutsStartupError` (thinking mode on, rollout logprobs leaked into
   ``old_log_probs``, NaN with ``abort_on_nan``) is printed once and exits with code 2.

Run (from the repo root, inside an allocation)::

    python -m mixed_cuts.main --config-name smoke hydra.searchpath=[pkg://verl.trainer.config] \\
        paths.run_dir=... paths.model_dir=... paths.data_dir=... run_name=smoke-s42
"""

from __future__ import annotations

import logging
import os
import random
import sys

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig

from mixed_cuts.errors import MixedCutsStartupError

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "train"
)


def _unwrap(cls):
    return getattr(cls, "__ray_actor_class__", cls)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _make_task_runner():
    from verl.trainer.main_ppo import TaskRunnerV1

    class _MixedCutsTaskRunnerBase(_unwrap(TaskRunnerV1)):  # type: ignore[misc,valid-type]
        """``TaskRunnerV1`` that registers our trainer / agent loop before running."""

        def run(self, config: DictConfig):
            import mixed_cuts.agent_loop  # noqa: F401  (imported for its side effects / early failure)
            import mixed_cuts.trainer  # noqa: F401  registers "mixed_cuts_sync"

            seed_everything(int(config.get("seed", 42)))
            logger.info(
                "Mixed-CUTS task runner: trainer_mode=%s seed=%s",
                config.trainer.v1.trainer_mode,
                config.get("seed"),
            )
            return super().run(config)

    return ray.remote(_MixedCutsTaskRunnerBase)


@hydra.main(config_path=CONFIG_DIR, config_name="base_grpo", version_base=None)
def main(config: DictConfig) -> None:
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if not config.trainer.get("use_v1", True):
        raise ValueError("Mixed-CUTS requires verl's V1 trainer (trainer.use_v1=true)")
    seed_everything(int(config.get("seed", 42)))
    try:
        run_ppo(config, task_runner_class=_make_task_runner())
    except Exception as e:  # noqa: BLE001 - Ray wraps the actor's exception; unwrap by name
        cause = getattr(e, "cause", None)
        if (
            isinstance(e, MixedCutsStartupError)
            or isinstance(cause, MixedCutsStartupError)
            or (
                "MixedCutsStartupError" in str(e)
                or "ThinkingModeError" in str(e)
                or "StabilityAbort" in str(e)
            )
        ):
            print(
                "\n" + "=" * 88 + f"\nMIXED-CUTS STARTUP CHECK FAILED\n{e}\n" + "=" * 88 + "\n",
                file=sys.stderr,
            )
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
