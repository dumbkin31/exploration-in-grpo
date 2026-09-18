"""Sampling backends for the eval harness. vLLM offline is the real one; the interface is a
plain callable so tests can substitute a stub."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

Messages = Sequence[dict[str, str]]
# generate(list_of_conversations, n_samples, sampling_cfg) -> list (per conversation) of n texts
GenerateFn = Callable[[Sequence[Messages], int, dict[str, Any]], list[list[str]]]


class VllmGenerator:
    """Wraps ``vllm.LLM`` for chat-formatted, n-sample generation with fp16 on sm_75."""

    def __init__(
        self, model_path: str, engine_cfg: dict[str, Any], chat_template_kwargs: dict[str, Any] | None = None
    ):
        from vllm import LLM  # lazy: the harness's pure parts must import without vLLM

        self.model_path = model_path
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        kwargs = dict(engine_cfg)
        logger.info("Starting vLLM: model=%s %s", model_path, kwargs)
        self.llm = LLM(model=model_path, **kwargs)

    def __call__(
        self, conversations: Sequence[Messages], n: int, sampling_cfg: dict[str, Any]
    ) -> list[list[str]]:
        from vllm import SamplingParams

        sp = SamplingParams(n=n, **sampling_cfg)
        outputs = self.llm.chat(
            [list(c) for c in conversations],
            sampling_params=sp,
            use_tqdm=True,
            chat_template_kwargs=self.chat_template_kwargs or None,
        )
        return [[o.text for o in req.outputs] for req in outputs]
