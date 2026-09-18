"""Training entry point: verl's ``main_ppo`` with the Mixed-CUTS pieces registered.

This mirrors ``verl/trainer/main_ppo.py`` (v0.9.0). The only differences:

1. the Hydra primary config is ours (``configs/train/<name>.yaml``), which layers on verl's
   ``ppo_trainer`` through ``hydra.searchpath: [pkg://verl.trainer.config]``;
2. the Ray task runner imports :mod:`mixed_cuts.trainer` (which registers the
   ``mixed_cuts_sync`` trainer) *inside the actor process* before verl looks the trainer up.
   Registration must happen in that process, not just in the driver.

Run (from the repo root, inside an allocation)::

    python -m mixed_cuts.main --config-name smoke paths.run_dir=/scratch/... paths.model_dir=...
"""

from __future__ import annotations

import logging
import os

import hydra
import ray
from omegaconf import DictConfig
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "train"
)


def _unwrap(cls):
    return getattr(cls, "__ray_actor_class__", cls)


class _MixedCutsTaskRunnerBase(_unwrap(TaskRunnerV1)):  # type: ignore[misc,valid-type]
    """``TaskRunnerV1`` that registers our trainer / agent loop before running."""

    def run(self, config: DictConfig):
        import mixed_cuts.agent_loop  # noqa: F401  (imported for its side effects / early failure)
        import mixed_cuts.trainer  # noqa: F401  registers "mixed_cuts_sync"

        logger.info("Mixed-CUTS task runner: trainer_mode=%s", config.trainer.v1.trainer_mode)
        return super().run(config)


MixedCutsTaskRunner = ray.remote(_MixedCutsTaskRunnerBase)


@hydra.main(config_path=CONFIG_DIR, config_name="base_grpo", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if not config.trainer.get("use_v1", True):
        raise ValueError("Mixed-CUTS requires verl's V1 trainer (trainer.use_v1=true)")
    run_ppo(config, task_runner_class=MixedCutsTaskRunner)


if __name__ == "__main__":
    main()
