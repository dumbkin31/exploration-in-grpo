"""Mixed-CUTS group planning: which rollouts of a prompt are standard and which are CUTS.

A GRPO group for one prompt has ``n = n_std + n_cuts`` rollouts. The first ``n_std`` sessions
sample normally ("exploitative"); the remaining ``n_cuts`` carry CUTS parameters in their
``extra_args`` ("exploratory"). All of them share the prompt's ``uid``, so verl's GRPO
advantage (which groups by ``uid``) normalises the *combined* group, exactly as Mixed-CUTS
prescribes.

This module is pure Python (no verl, no vLLM) so the decision logic can be unit-tested on a
laptop. :mod:`mixed_cuts.agent_loop` is the thin verl wrapper that calls :func:`plan_group`.

Invariant that makes ``n_cuts = 0`` reproduce vanilla GRPO: every standard session receives a
plain copy of the base sampling params, byte-for-byte what verl's own ``_run_prompt`` builds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from cuts.config import CutsParams

STD = "std"
CUTS = "cuts"
ROLLOUT_KINDS = (STD, CUTS)


@dataclass(frozen=True)
class SessionSpec:
    """One rollout of a group."""

    session_id: int
    rollout_kind: str
    sampling_params: dict[str, Any]


@dataclass(frozen=True)
class MixedCutsConfig:
    """The ``mixed_cuts:`` block of a training config (see configs/train/base_grpo.yaml)."""

    enabled: bool = True
    n_std: int = 8
    n_cuts: int = 8
    group_size: int | None = None
    """G, the GRPO group size. Defaults to ``n_std + n_cuts``; when given it must equal it (D4)."""
    cuts: CutsParams = field(default_factory=CutsParams)
    dump_samples_per_step: int = 4
    """How many whole groups to dump to disk each step for manual reading."""

    def __post_init__(self) -> None:
        if self.n_std < 0 or self.n_cuts < 0:
            raise ValueError(f"mixed_cuts.n_std / n_cuts must be >= 0, got {self.n_std} / {self.n_cuts}")
        if self.enabled and self.n_std + self.n_cuts == 0:
            raise ValueError("mixed_cuts.enabled but n_std + n_cuts == 0")
        if self.group_size is None:
            object.__setattr__(self, "group_size", self.n_std + self.n_cuts)  # frozen dataclass
        # D4: both arms must cost the same generation budget. Vanilla GRPO is n_std = G, n_cuts = 0.
        if self.enabled and self.n_std + self.n_cuts != self.group_size:
            raise ValueError(f"budget mismatch: {self.n_std} + {self.n_cuts} != G={self.group_size}")

    @property
    def n(self) -> int:
        return self.n_std + self.n_cuts

    @property
    def active(self) -> bool:
        """True iff some rollouts will actually use CUTS."""
        return self.enabled and self.n_cuts > 0

    def validate_against_rollout_n(self, rollout_n: int) -> None:
        """``actor_rollout_ref.rollout.n`` must equal G when CUTS is on (verl uses rollout.n everywhere)."""
        if self.enabled and self.group_size != rollout_n:
            raise ValueError(
                f"budget mismatch: mixed_cuts.group_size (= n_std + n_cuts = {self.group_size}) != "
                f"actor_rollout_ref.rollout.n = {rollout_n}; set rollout.n: ${{mixed_cuts.group_size}}"
            )

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> MixedCutsConfig:
        """Build from a plain dict or an OmegaConf ``DictConfig`` (values are copied out)."""
        if cfg is None:
            return cls(enabled=False, n_std=0, n_cuts=0)
        d = {k: cfg[k] for k in cfg.keys()}  # works for dict and DictConfig
        cuts_raw = d.pop("cuts", None) or {}
        cuts_kwargs = {k: cuts_raw[k] for k in cuts_raw.keys()}
        # Bookkeeping fields are filled per request by plan_group; a config may set stats_dir.
        for k in ("uid", "session_id", "step"):
            cuts_kwargs.pop(k, None)
        if "stats_dir" in cuts_kwargs and cuts_kwargs["stats_dir"] is not None:
            cuts_kwargs["stats_dir"] = str(cuts_kwargs["stats_dir"])
        return cls(cuts=CutsParams(**cuts_kwargs), **d)


def plan_group(
    base_sampling_params: Mapping[str, Any],
    n: int,
    config: MixedCutsConfig,
    *,
    uid: str,
    step: int | None,
    force_standard: bool = False,
) -> list[SessionSpec]:
    """Decide the kind and sampling params of each of the ``n`` rollouts of one prompt.

    Args:
        base_sampling_params: what verl would give every rollout (temperature, top_p, ...).
        n: group size requested for this prompt (``rollout.n``, or a per-prompt override).
        config: the ``mixed_cuts`` block.
        uid: the prompt's group id (stamped into the CUTS params for the |S_t| statistics).
        step: training global step (same purpose).
        force_standard: validation rollouts and greedy (``do_sample=False``) rollouts are
            always standard, regardless of the config.

    Returns:
        ``n`` :class:`SessionSpec` in session order: standard sessions first, CUTS sessions last.
    """
    if force_standard or not config.active:
        # Exactly what verl's AgentLoopWorkerTQ._run_prompt does: one copy of the params per session.
        return [SessionSpec(i, STD, dict(base_sampling_params)) for i in range(n)]

    if n != config.group_size:
        raise ValueError(
            f"prompt {uid!r} asks for n={n} rollouts but mixed_cuts is configured for "
            f"G = n_std + n_cuts = {config.group_size}; per-prompt __rollout_n__ overrides are not supported"
        )

    specs = [SessionSpec(i, STD, dict(base_sampling_params)) for i in range(config.n_std)]
    for session_id in range(config.n_std, config.n):
        params = replace(config.cuts, uid=str(uid), session_id=session_id, step=step)
        sampling_params = dict(base_sampling_params)
        # Preserve anything verl already put in extra_args (e.g. kv_transfer_params).
        sampling_params["extra_args"] = {
            **(base_sampling_params.get("extra_args") or {}),
            **params.to_extra_args(),
        }
        # D2: vLLM's own truncation must not re-narrow the already-uniform candidate set. Its
        # top-k/top-p run AFTER the logits processor, so with top_p < 1 they would drop survivors.
        sampling_params["top_p"] = 1.0
        sampling_params["top_k"] = -1
        specs.append(SessionSpec(session_id, CUTS, sampling_params))
    return specs
