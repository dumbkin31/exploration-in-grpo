"""Mixed standard / CUTS rollout scheduling inside verl's rollout worker.

Hook point (verl v0.9.0, ``verl/trainer/ppo/v1/agent_loop_tq.py``): ``AgentLoopWorkerTQ._run_prompt``
is the coroutine that turns one prompt into its ``n`` rollout sessions. It builds one
``run_sampling_params`` dict and spawns ``n`` ``_run_agent_loop`` tasks with ``session_id=i``.
We override only that method: the ``n`` sessions are planned by
:func:`mixed_cuts.scheduler.plan_group`, so the last ``n_cuts`` of them carry CUTS parameters in
``extra_args`` and every session is tagged with ``rollout_kind``. Everything downstream is
untouched:

* the tag rides along in ``**kwargs`` -> ``_agent_loop_postprocess`` -> ``field.update(kwargs)``,
  so it is stored in TransferQueue next to ``uid`` and can be read back by the trainer;
* all sessions share the prompt ``uid``, so verl's GRPO advantage normalises the mixed group;
* verl's ``rollout.n`` stays the group size, so batch-size validation is unaffected.

Ray detail: verl decorates ``AgentLoopWorkerTQ`` with ``@ray.remote`` at definition time, and
Ray forbids inheriting from an actor class. Like verl itself does
(``verl/single_controller/ray/base.py``), we subclass the undecorated class exposed as
``__ray_actor_class__`` and re-wrap the subclass with ``ray.remote``.

Plugged in with one config line::

    actor_rollout_ref:
      rollout:
        agent:
          agent_loop_manager_class: mixed_cuts.agent_loop.MixedCutsAgentLoopManagerTQ
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray
import transfer_queue as tq
from verl.trainer.ppo.v1 import agent_loop_tq as _verl_tq

from mixed_cuts.scheduler import MixedCutsConfig, plan_group

logger = logging.getLogger(__name__)


def _unwrap_ray_actor_class(cls: Any) -> type:
    """Return the plain Python class behind a ``@ray.remote``-decorated class (verl's own trick)."""
    if hasattr(cls, "__ray_actor_class__"):
        return cls.__ray_actor_class__
    meta = getattr(cls, "__ray_metadata__", None)
    if meta is not None and hasattr(meta, "modified_class"):
        return getattr(meta.modified_class, "__ray_actor_class__", meta.modified_class)
    return cls


_BaseWorkerTQ: type = _unwrap_ray_actor_class(_verl_tq.AgentLoopWorkerTQ)


class MixedCutsAgentLoopWorkerTQBase(_BaseWorkerTQ):  # type: ignore[misc,valid-type]
    """``AgentLoopWorkerTQ`` whose ``_run_prompt`` emits ``n_std`` standard + ``n_cuts`` CUTS sessions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_mixed_cuts()

    def _init_mixed_cuts(self) -> None:
        """Parse and validate the ``mixed_cuts`` block (separate so tests can build bare instances)."""
        self.mixed_cuts_config = MixedCutsConfig.from_mapping(self.config.get("mixed_cuts", None))
        self.mixed_cuts_config.validate_against_rollout_n(int(self.config.actor_rollout_ref.rollout.n))
        logger.info(
            "Mixed-CUTS scheduling: enabled=%s n_std=%d n_cuts=%d cuts=%s",
            self.mixed_cuts_config.enabled,
            self.mixed_cuts_config.n_std,
            self.mixed_cuts_config.n_cuts,
            self.mixed_cuts_config.cuts,
        )

    async def _run_prompt(
        self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False
    ) -> None:
        """Spawn the ``n`` sessions of one prompt; mirrors verl v0.9.0 line for line except for the planning.

        verl's version (``agent_loop_tq.py:107-148``) creates ``n`` tasks with the *same*
        ``run_sampling_params`` dict. With ``n_cuts == 0`` (or CUTS disabled, or a validation /
        greedy prompt) :func:`plan_group` hands every session an equal copy of that dict, so the
        behaviour is identical to vanilla GRPO. This is asserted by ``tests/test_agent_loop_verl.py``.
        """
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks: list[asyncio.Task[Any]] = []
        try:
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                _verl_tq.apply_greedy_sampling_params(run_sampling_params)

            # --- the only Mixed-CUTS change: decide std vs CUTS per session -------------------
            specs = plan_group(
                run_sampling_params,
                int(n),
                self.mixed_cuts_config,
                uid=str(uid),
                step=trajectory.get("step"),
                force_standard=bool(trajectory["validate"]) or not do_sample,
            )
            for spec in specs:
                task = asyncio.create_task(
                    self._run_agent_loop(
                        spec.sampling_params,
                        trajectory=trajectory,
                        trace=trace,
                        session_id=spec.session_id,
                        rollout_kind=spec.rollout_kind,
                        **prompt,
                    )
                )
                tasks.append(task)
            # ----------------------------------------------------------------------------------

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _verl_tq._settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _verl_tq._settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})


# The Ray actor class verl's manager instantiates (same plain ``@ray.remote`` as verl uses).
MixedCutsAgentLoopWorkerTQ = ray.remote(MixedCutsAgentLoopWorkerTQBase)


class MixedCutsAgentLoopManagerTQ(_verl_tq.AgentLoopManagerTQ):
    """``AgentLoopManagerTQ`` that spawns :class:`MixedCutsAgentLoopWorkerTQ` workers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # AgentLoopManagerTQ.__init__ assigns its own worker class first; workers are only created
        # later in create() -> _init_agent_loop_workers(), so overriding after super() is enough.
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = MixedCutsAgentLoopWorkerTQ
        # Fail at start-up, not at the first prompt, if n_std + n_cuts != rollout.n.
        MixedCutsConfig.from_mapping(self.config.get("mixed_cuts", None)).validate_against_rollout_n(
            int(self.config.actor_rollout_ref.rollout.n)
        )
