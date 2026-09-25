"""Diagnostics and safety hooks: a ``PPOTrainerSync`` subclass for Mixed-CUTS.

Hook points (verl v0.9.0, ``verl/trainer/ppo/v1/trainer_base.py``):

* ``_compute_old_log_prob``  (line ~1479): the actor recomputes ``old_log_probs``. BEFORE the first
  call we assert Qwen3 is in non-thinking mode (D2) on token ids; AFTER it we run the D1 detector:
  the fraction of CUTS tokens whose ``old_log_probs`` sits on the ``-log(j)`` lattice must be
  small, otherwise rollout-engine logprobs leaked in and the method would be cancelled. Both raise
  :class:`MixedCutsStartupError` before any update.
* ``on_sample_begin/on_sample_end``, ``_compute_ref_log_prob``, ``_update_actor``: phase markers
  for the memory profiler (``phases.jsonl``). ``on_sample_end`` MUST call super (it sleeps vLLM).
* ``_compute_metrics`` (line ~1713): after verl's metrics we add the D5/D6 diagnostics, the |S_t|
  summary (one step late, see cuts.stats), ACR_100, the fp16 stability watch (Task D), and mirror
  the full metrics dict into ``metrics.jsonl`` (the primary log; W&B is offline on Ada).
* ``_log_rollout_data``: a few whole groups (std + CUTS siblings) per step instead of everything.

Data comes from TransferQueue the way verl reads it (``tq.kv_batch_get(select_fields=...)``);
missing fields are dropped silently by TransferQueue, so every access is guarded.
Registered as ``trainer.v1.trainer_mode: mixed_cuts_sync``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer
from verl.utils.debug import marked_timer

from cuts.stats import list_stats_files, read_cuts_stats, summarize_cuts_stats
from mixed_cuts.diagnostics import RunningMean, compute_group_diagnostics, uniform_logprob_fraction
from mixed_cuts.errors import MixedCutsStartupError
from mixed_cuts.jsonl_logger import JsonlMetricsLogger
from mixed_cuts.phases import PhaseLog
from mixed_cuts.scheduler import CUTS, STD, MixedCutsConfig
from mixed_cuts.stability import StabilityConfig, StabilityWatch
from mixed_cuts.thinking import check_non_thinking

logger = logging.getLogger(__name__)

TRAINER_NAME = "mixed_cuts_sync"
ACR_KEY = "mixed_cuts/advantage_collapse_rate"
ACR_100_KEY = "mixed_cuts/ACR_100"
UNIFORM_LP_KEY = "cuts/frac_old_logprob_uniform_like_k2plus"
UNIFORM_LP_ANY_KEY = "cuts/frac_old_logprob_uniform_like_any"
UNIFORM_LP_ALARM = 0.5


# ----------------------------------------------------------------------------- pure helpers
def _padded(t: Any, pad: float = 0.0) -> torch.Tensor:
    return t.to_padded_tensor(pad) if getattr(t, "is_nested", False) else t


def _sequence_scores(rm_scores: Any) -> list[float]:
    """Sum token-level scores to one scalar per rollout; handles nested and padded tensors."""
    return _padded(rm_scores).float().sum(dim=-1).tolist()


def _sequence_lengths(response_mask: Any) -> list[int] | None:
    """Response length in tokens per rollout, from ``response_mask`` (1 = generated token).

    Works for a nested (ragged) tensor and for a right-padded ``[B, T]`` tensor. It must NOT be
    computed from ``responses.shape[-1]``: that is the padded width, identical for every row.
    """
    if response_mask is None:
        return None
    if getattr(response_mask, "is_nested", False):
        try:
            return [int(t.sum()) for t in response_mask.unbind()]
        except Exception:  # noqa: BLE001 - layout-dependent API
            return response_mask.offsets().diff().tolist()
    if isinstance(response_mask, torch.Tensor):
        return response_mask.to(torch.int64).sum(dim=-1).tolist()
    return None


def _non_tensor_list(data: Any, key: str, n: int, default: Any) -> list[Any]:
    """Read a non-tensor field (NonTensorStack) as a plain list, with a fallback when absent."""
    if key not in data.keys():
        return [default] * n
    value = data[key]
    return list(value.tolist()) if hasattr(value, "tolist") else list(value)


def _id_lists(nested: Any) -> list[list[int]]:
    if getattr(nested, "is_nested", False):
        return [t.tolist() for t in nested.unbind()]
    return [row.tolist() for row in nested]


# ----------------------------------------------------------------------------- the trainer
@register_trainer(TRAINER_NAME)
class MixedCutsPPOTrainerSync(PPOTrainerSync):
    """verl's synchronous V1 PPO trainer plus Mixed-CUTS diagnostics and startup assertions."""

    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.mixed_cuts_config = MixedCutsConfig.from_mapping(config.get("mixed_cuts", None))
        self.mixed_cuts_config.validate_against_rollout_n(int(config.actor_rollout_ref.rollout.n))
        checks = (config.get("mixed_cuts", None) or {}).get("checks", None) or {}
        self._check_non_thinking = bool(checks.get("assert_non_thinking", True))
        self._check_recomputed = bool(checks.get("assert_recomputed_logprobs", True))

        run_dir = config.get("paths", {}).get("run_dir", None) or config.trainer.default_local_dir
        self._mc_run_dir = str(run_dir)
        self._mc_jsonl = JsonlMetricsLogger(os.path.join(self._mc_run_dir, "metrics.jsonl"))
        self._phases = PhaseLog(os.path.join(self._mc_run_dir, "phases.jsonl"))
        self._mc_stats_dir = self.mixed_cuts_config.cuts.stats_dir
        # Files from a previous incarnation of this run (killed mid-step) must not be re-read.
        self._stale_stats_files = list_stats_files(self._mc_stats_dir) if self._mc_stats_dir else set()

        self._acr_100 = RunningMean(max_steps=100)
        self._stability = StabilityWatch(StabilityConfig.from_mapping(config.get("stability", None)))
        previous = self._mc_jsonl.read()  # empty on a fresh run; the history after a resume
        self._acr_100.rebuild(previous, ACR_KEY)
        self._stability.rebuild(previous)

        self._first_old_logprob_done = False
        self._pending_metrics: dict[str, float] = {}
        logger.info(
            "Mixed-CUTS trainer: run dir %s (metrics.jsonl, phases.jsonl), cuts stats %s (%d stale files ignored), "
            "resumed history: %d steps",
            self._mc_run_dir,
            self._mc_stats_dir,
            len(self._stale_stats_files),
            len(previous),
        )

    # ------------------------------------------------------------------ phase markers
    def on_sample_begin(self):
        self._phases.begin(self.global_steps, "rollout")
        return super().on_sample_begin()

    def on_sample_end(self):
        self._phases.end(self.global_steps, "rollout")
        return super().on_sample_end()  # sleeps the vLLM replicas; never skip

    def _compute_ref_log_prob(self, batch, metrics: dict):
        with self._phases.phase(self.global_steps, "ref_log_prob"):
            return super()._compute_ref_log_prob(batch, metrics)

    def _update_actor(self, batch, metrics: dict):
        with self._phases.phase(self.global_steps, "update_actor"):
            return super()._update_actor(batch, metrics)

    # ------------------------------------------------------------------ D1 + D2 at the first step
    def _compute_old_log_prob(self, batch, metrics: dict):
        first = not self._first_old_logprob_done
        if first and self._check_non_thinking:
            self._assert_non_thinking(batch)
        with self._phases.phase(self.global_steps, "old_log_prob"):
            batch = super()._compute_old_log_prob(batch, metrics)
        if self.mixed_cuts_config.active and self._check_recomputed:
            self._pending_metrics.update(self._recomputed_logprob_check(batch, raise_on_alarm=first))
        self._first_old_logprob_done = True
        return batch

    def _assert_non_thinking(self, batch) -> None:
        """D2: fail loudly if any response of the first batch carries <think>/</think> tokens."""
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
        )
        if "responses" not in data.keys():
            logger.warning("non-thinking check skipped: 'responses' not in TransferQueue")
            return
        prompts = _id_lists(data["prompts"]) if "prompts" in data.keys() else None
        stats = check_non_thinking(
            self.tokenizer, prompts, _id_lists(data["responses"]), where="in the first training batch"
        )
        self._pending_metrics.update(stats)
        if stats.get("mixed_cuts/frac_prompts_without_empty_think_block", 0.0) > 0:
            logger.warning(
                "%.0f%% of prompts do not end with the empty <think></think> block; check "
                "data.apply_chat_template_kwargs.enable_thinking=false reaches the chat template",
                100 * stats["mixed_cuts/frac_prompts_without_empty_think_block"],
            )
        logger.info("non-thinking mode confirmed on %d responses", len(data["responses"]))

    def _recomputed_logprob_check(self, batch, *, raise_on_alarm: bool) -> dict[str, float]:
        """D1: old_log_probs must be pi_theta_old (actor recompute), never the engine's log(1/|S_t|)."""
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["old_log_probs", "response_mask", "rollout_kind"],
        )
        if "old_log_probs" not in data.keys() or "response_mask" not in data.keys():
            logger.warning("D1 check skipped: old_log_probs/response_mask not in TransferQueue")
            return {}
        n = len(batch.keys)
        kinds = _non_tensor_list(data, "rollout_kind", n, STD)
        is_cuts = [k == CUTS for k in kinds]
        lp = _padded(data["old_log_probs"], 0.0)
        mask = _padded(data["response_mask"], 0)
        k, t_warm = self.mixed_cuts_config.cuts.k, self.mixed_cuts_config.cuts.t_warm
        frac_k2, n_tok = uniform_logprob_fraction(lp, mask, is_cuts, k=k, t_warm=t_warm, k_min=2)
        frac_any, _ = uniform_logprob_fraction(lp, mask, is_cuts, k=k, t_warm=t_warm, k_min=1)
        out = {UNIFORM_LP_KEY: frac_k2, UNIFORM_LP_ANY_KEY: frac_any, "cuts/n_tokens_checked": float(n_tok)}
        if raise_on_alarm and n_tok > 0 and frac_k2 > UNIFORM_LP_ALARM:
            raise MixedCutsStartupError(
                f"old_log_probs look like the rollout engine's log(1/|S_t|) on {100 * frac_k2:.1f}% of CUTS tokens "
                f"(> {100 * UNIFORM_LP_ALARM:.0f}%). The actor must recompute them: check "
                "algorithm.rollout_correction.bypass_mode=false and rollout.calculate_log_probs=false (D1)."
            )
        logger.info(
            "D1 check: %.2f%% of %d CUTS tokens on the -log(j>=2) lattice (alarm at %.0f%%)",
            100 * frac_k2,
            n_tok,
            100 * UNIFORM_LP_ALARM,
        )
        return out

    # ------------------------------------------------------------------ per-step metrics
    def _compute_metrics(self, batch, metrics: dict, timing_raw: dict, global_steps: int, epoch: int) -> None:
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        try:
            with marked_timer("mixed_cuts_diagnostics", timing_raw, color="magenta"):
                metrics.update(self._mixed_cuts_metrics(batch, global_steps))
        except Exception:  # noqa: BLE001 - diagnostics must never kill a run
            logger.exception("Mixed-CUTS diagnostics failed at step %d", global_steps)
        metrics.update(self._pending_metrics)
        self._pending_metrics = {}
        acr = metrics.get(ACR_KEY)
        if acr is not None:
            value = self._acr_100.update(global_steps, float(acr))
            if value is not None:
                metrics[ACR_100_KEY] = value
        metrics.update(self._stability.check(metrics, global_steps))  # may raise StabilityAbort (config)
        try:
            self._mc_jsonl.log(metrics, step=global_steps)
        except Exception:  # noqa: BLE001
            logger.exception("metrics.jsonl write failed at step %d", global_steps)

    def _mixed_cuts_metrics(self, batch, global_steps: int) -> dict[str, float]:
        non_padding = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "rm_scores", "response_mask", "rollout_kind"],
        )
        n = len(batch.keys)
        uids = _non_tensor_list(data, "uid", n, None)
        kinds = _non_tensor_list(data, "rollout_kind", n, STD)
        scores = _sequence_scores(data["rm_scores"])
        lengths = _sequence_lengths(data["response_mask"]) if "response_mask" in data.keys() else None

        keep = np.flatnonzero(non_padding) if non_padding.size == n else np.arange(n)
        out = compute_group_diagnostics(
            [uids[i] for i in keep],
            [scores[i] for i in keep],
            [kinds[i] for i in keep],
            None if lengths is None else [lengths[i] for i in keep],
            group_size=self.mixed_cuts_config.group_size,
        )
        if self._mc_stats_dir and self.mixed_cuts_config.active:
            # One-step lag: a request's stats are flushed by the engine's NEXT forward pass, so
            # step N's records are complete once step N+1's generation has started.
            stats_step = global_steps - 1
            if stats_step >= 1:
                records = read_cuts_stats(self._mc_stats_dir, step=stats_step, ignore=self._stale_stats_files)
                out.update(summarize_cuts_stats(records))
                out["cuts/stats_step"] = float(stats_step)
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
        pad = self.tokenizer.pad_token_id
        prompts = _padded(data["prompts"], pad)
        responses = _padded(data["responses"], pad)
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
