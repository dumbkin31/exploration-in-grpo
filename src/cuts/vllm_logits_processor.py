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
import os

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


def _tensor_parallel_rank() -> int:
    """TP rank of this engine process, or 0 when vLLM's process groups are not initialised."""
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - not initialised (tests, single process) -> rank 0
        return 0


class CutsLogitsProcessor(LogitsProcessor):
    """Applies CUTS to the rows of the persistent batch whose request asked for it.

    With tensor parallelism every TP rank runs the sampler (and therefore this processor) on the
    full batch, so per-request statistics would be written once per rank. Only TP rank 0 owns a
    writer; the other ranks track state identically but discard the records.
    """

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:  # noqa: ARG002
        self.device = device
        self.tp_rank = _tensor_parallel_rank()
        self._writer = CutsStatsWriter(tp_rank=self.tp_rank) if self.tp_rank == 0 else None
        self._state = CutsBatchState(on_request_finished=self._writer.write if self._writer else None)
        logger.info(
            "CutsLogitsProcessor initialised on %s (pid %d, tp_rank %d, writes_stats=%s)",
            device,
            os.getpid(),
            self.tp_rank,
            self._writer is not None,
        )

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
        if self._writer is not None:
            self._writer.close()
