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
        self._checked_thinking = False
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
        if not self._checked_thinking:
            # D2: fail loudly the first time if Qwen3 ran in thinking mode (vLLM silently drops
            # unknown chat_template_kwargs, so this is the only reliable check).
            from mixed_cuts.thinking import check_non_thinking

            check_non_thinking(
                self.llm.get_tokenizer(),
                [list(req.prompt_token_ids or []) for req in outputs],
                [list(o.token_ids) for req in outputs for o in req.outputs],
                where="in eval generation",
            )
            self._checked_thinking = True
        return [[o.text for o in req.outputs] for req in outputs]
