"""Diagnostics hook: a ``PPOTrainerSync`` subclass that logs Mixed-CUTS metrics every step.

Hook points (verl v0.9.0, ``verl/trainer/ppo/v1/trainer_base.py``):

* ``_compute_metrics(batch, metrics, timing_raw, global_steps, epoch)`` is called once per step
  after training, right before ``self.logger.log(...)``. We call verl's version and then add
  the group-level reward diagnostics (:mod:`mixed_cuts.diagnostics`) and the |S_t| summary
  (:mod:`cuts.stats`), and mirror the full metrics dict into a local ``metrics.jsonl``.
* ``_log_rollout_data(batch, timing_raw, rollout_data_dir)`` dumps generations when
  ``trainer.rollout_data_dir`` is set. We replace verl's dump-everything with a small sample of
  *whole groups* tagged with ``rollout_kind`` so standard and CUTS siblings read side by side.

Data comes from TransferQueue exactly the way verl reads it (``tq.kv_batch_get(select_fields=...)``):
``uid`` and ``rollout_kind`` are non-tensor fields, ``rm_scores`` / ``responses`` are nested
(ragged) tensors. Registered as ``trainer.v1.trainer_mode: mixed_cuts_sync``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer
from verl.utils.py_functional import marked_timer

from cuts.stats import read_cuts_stats, summarize_cuts_stats
from mixed_cuts.diagnostics import compute_group_diagnostics
from mixed_cuts.jsonl_logger import JsonlMetricsLogger
from mixed_cuts.scheduler import STD, MixedCutsConfig

logger = logging.getLogger(__name__)

TRAINER_NAME = "mixed_cuts_sync"


def _sequence_scores(rm_scores: Any) -> list[float]:
    """Sum token-level scores to one scalar per rollout; handles nested and padded tensors."""
    if getattr(rm_scores, "is_nested", False):
        rm_scores = rm_scores.to_padded_tensor(0.0)
    return rm_scores.float().sum(dim=-1).tolist()


def _sequence_lengths(responses: Any) -> list[int] | None:
    if getattr(responses, "is_nested", False):
        try:
            return responses.offsets().diff().tolist()
        except Exception:  # noqa: BLE001 - layout-dependent API
            return [int(t.numel()) for t in responses.unbind()]
    if isinstance(responses, torch.Tensor):
        return [int(responses.shape[-1])] * int(responses.shape[0])
    return None


def _non_tensor_list(data: Any, key: str, n: int, default: Any) -> list[Any]:
    """Read a non-tensor field (NonTensorStack) as a plain list, with a fallback when absent."""
    if key not in data.keys():
        return [default] * n
    value = data[key]
    return list(value.tolist()) if hasattr(value, "tolist") else list(value)


@register_trainer(TRAINER_NAME)
class MixedCutsPPOTrainerSync(PPOTrainerSync):
    """verl's synchronous V1 PPO trainer plus Mixed-CUTS diagnostics."""

    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.mixed_cuts_config = MixedCutsConfig.from_mapping(config.get("mixed_cuts", None))
        run_dir = config.get("paths", {}).get("run_dir", None) or config.trainer.default_local_dir
        self._mc_run_dir = str(run_dir)
        self._mc_jsonl = JsonlMetricsLogger(os.path.join(self._mc_run_dir, "metrics.jsonl"))
        self._mc_stats_dir = self.mixed_cuts_config.cuts.stats_dir
        logger.info(
            "Mixed-CUTS trainer: metrics.jsonl at %s, cuts stats dir %s",
            self._mc_jsonl.path,
            self._mc_stats_dir,
        )

    # ------------------------------------------------------------------ per-step metrics
    def _compute_metrics(self, batch, metrics: dict, timing_raw: dict, global_steps: int, epoch: int) -> None:
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        try:
            with marked_timer("mixed_cuts_diagnostics", timing_raw, color="magenta"):
                metrics.update(self._mixed_cuts_metrics(batch, global_steps))
        except Exception:  # noqa: BLE001 - diagnostics must never kill a run
            logger.exception("Mixed-CUTS diagnostics failed at step %d", global_steps)
        try:
            self._mc_jsonl.log(metrics, step=global_steps)
        except Exception:  # noqa: BLE001
            logger.exception("metrics.jsonl write failed at step %d", global_steps)

    def _mixed_cuts_metrics(self, batch, global_steps: int) -> dict[str, float]:
        non_padding = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "rm_scores", "responses", "rollout_kind"],
        )
        n = len(batch.keys)
        uids = _non_tensor_list(data, "uid", n, None)
        kinds = _non_tensor_list(data, "rollout_kind", n, STD)
        scores = _sequence_scores(data["rm_scores"])
        lengths = _sequence_lengths(data["responses"])

        keep = np.flatnonzero(non_padding) if non_padding.size == n else np.arange(n)
        out = compute_group_diagnostics(
            [uids[i] for i in keep],
            [scores[i] for i in keep],
            [kinds[i] for i in keep],
            None if lengths is None else [lengths[i] for i in keep],
        )
        if self._mc_stats_dir:
            out.update(summarize_cuts_stats(read_cuts_stats(self._mc_stats_dir, step=global_steps)))
        return out

    # ------------------------------------------------------------------ rollout samples
    def _log_rollout_data(self, batch, timing_raw: dict, rollout_data_dir: str) -> None:
        """Dump ``mixed_cuts.dump_samples_per_step`` whole groups (std + CUTS siblings) as JSONL."""
        try:
            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                self._dump_sampled_groups(batch, rollout_data_dir)
        except Exception:  # noqa: BLE001
            logger.exception("rollout dump failed; falling back to verl's dump")
            super()._log_rollout_data(batch, timing_raw, rollout_data_dir)

    def _dump_sampled_groups(self, batch, rollout_data_dir: str) -> None:
        fields = ["uid", "prompts", "responses", "rm_scores", "reward_model", "rollout_kind"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        n = len(batch.keys)
        uids = _non_tensor_list(data, "uid", n, None)
        kinds = _non_tensor_list(data, "rollout_kind", n, STD)
        scores = _sequence_scores(data["rm_scores"])
        reward_model = _non_tensor_list(data, "reward_model", n, {})
        gts = [rm.get("ground_truth") if isinstance(rm, dict) else None for rm in reward_model]

        # choose the first K groups by uid (deterministic), keep every rollout of those groups
        max_groups = int(self.mixed_cuts_config.dump_samples_per_step)
        chosen: list[Any] = []
        for u in uids:
            if u not in chosen:
                chosen.append(u)
            if len(chosen) >= max_groups:
                break
        chosen_set = set(chosen)
        rows = [i for i, u in enumerate(uids) if u in chosen_set]

        # sort: group, then standard before CUTS, then session id inside the key ({uid}_{session}_{idx})
        def sort_key(i: int):
            parts = batch.keys[i].rsplit("_", 2)
            session = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else 0
            return (chosen.index(uids[i]), 0 if kinds[i] == STD else 1, session)

        rows.sort(key=sort_key)
        if not rows:
            return

        prompts = data["prompts"]
        responses = data["responses"]
        pad = self.tokenizer.pad_token_id
        if getattr(prompts, "is_nested", False):
            prompts = prompts.to_padded_tensor(padding=pad)
        if getattr(responses, "is_nested", False):
            responses = responses.to_padded_tensor(padding=pad)
        inputs = [self.tokenizer.decode(prompts[i], skip_special_tokens=True) for i in rows]
        outputs = [self.tokenizer.decode(responses[i], skip_special_tokens=True) for i in rows]
        self._dump_generations(
            inputs=inputs,
            outputs=outputs,
            gts=[gts[i] for i in rows],
            scores=[scores[i] for i in rows],
            reward_extra_infos_dict={
                "uid": [batch.keys[i] for i in rows],
                "rollout_kind": [kinds[i] for i in rows],
            },
            dump_path=rollout_data_dir,
        )
