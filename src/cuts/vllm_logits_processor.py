"""CUTS as a vLLM custom logits processor (V1 sampler interface).

Interface verified against vLLM 0.24.0, ``vllm/v1/sample/logits_processor/interface.py``::

    class LogitsProcessor(ABC):
        @classmethod
        def validate_params(cls, sampling_params: SamplingParams): ...   # raise ValueError
        def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool): ...
        def apply(self, logits: torch.Tensor) -> torch.Tensor: ...      # may modify in place
        def is_argmax_invariant(self) -> bool: ...
        def update_state(self, batch_update: BatchUpdate | None): ...   # before each forward

How it is wired (all declarative, see configs/train/base_grpo.yaml):

* registered engine-wide through the ``vllm serve`` argument
  ``--logits-processors cuts.vllm_logits_processor:CutsLogitsProcessor`` (verl passes
  ``actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors`` straight through);
* activated per request by ``SamplingParams(extra_args={"cuts": {...}})``; requests without
  the key are standard requests and are never touched, so standard and CUTS rollouts share a
  batch.

Two vLLM facts that matter for correctness:

* vLLM 0.24.0's Model Runner V2 does not support custom logits processors and falls back to
  Model Runner V1 automatically (``vllm/config/vllm.py``). Never force
  ``VLLM_USE_V2_MODEL_RUNNER=1``.
* The V1 sampler applies non-argmax-invariant processors *before* temperature and top-k/top-p
  (``vllm/v1/sample/sampler.py``), so ``delta`` acts on temperature-1 probabilities exactly as in
  the paper, and the uniform output is unaffected by temperature.
"""

from __future__ import annotations

import logging

import torch

try:
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except ImportError as e:  # pragma: no cover - only hit outside the vLLM environment
    raise ImportError(
        "cuts.vllm_logits_processor needs vLLM (pinned 0.24.0). The operator itself lives in "
        "cuts.operator / cuts.state and works without vLLM."
    ) from e

from cuts.config import CutsParams
from cuts.state import CutsBatchState
from cuts.stats import CutsStatsWriter

logger = logging.getLogger(__name__)


class CutsLogitsProcessor(LogitsProcessor):
    """Applies CUTS to the rows of the persistent batch whose request asked for it."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:  # noqa: ARG002
        self.device = device
        self._writer = CutsStatsWriter()
        self._state = CutsBatchState(on_request_finished=self._writer.write)
        logger.info("CutsLogitsProcessor initialised on %s (pid %d)", device, __import__("os").getpid())

    # --- vLLM interface ---------------------------------------------------------------
    @classmethod
    def validate_params(cls, sampling_params) -> None:
        """Reject malformed ``extra_args["cuts"]`` at request time (vLLM turns this into a 400)."""
        try:
            CutsParams.from_extra_args(getattr(sampling_params, "extra_args", None))
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid CUTS request parameters: {e}") from e

    def is_argmax_invariant(self) -> bool:
        # Survivors all get logit 0, so the argmax is no longer the model's argmax.
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        self._state.apply_batch_update(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # In-place is allowed by the interface and avoids cloning [batch, vocab] every step.
        return self._state.apply(logits, inplace=True)

    # --- introspection (tests / debugging) --------------------------------------------
    @property
    def num_tracked_requests(self) -> int:
        return len(self._state)

    def close(self) -> None:
        self._state.clear()
        self._writer.close()
