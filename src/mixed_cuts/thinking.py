"""Assert that Qwen3 runs in NON-thinking mode (D2), on token ids.

With ``enable_thinking=False`` Qwen3's chat template appends an empty reasoning block
``<think>\\n\\n</think>\\n\\n`` to the END OF THE PROMPT (the assistant prefix). Two consequences:

* the response must not contain ``<think>`` / ``</think>`` (they are real tokens, ids
  151667 / 151668 for the Qwen3 tokenizer; resolved at runtime, never hardcoded);
* every prompt should end with that block, which proves the chat-template kwarg was honoured
  (vLLM silently drops unknown ``chat_template_kwargs``).

Used by the trainer (first step, before the first update), the eval generator (first batch) and
the tokenizer-only preflight.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


class ThinkingModeError(RuntimeError):
    """Raised when thinking-mode tags show up where non-thinking output was required."""


def think_token_ids(tokenizer: Any) -> tuple[int, int]:
    ids = tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
    unk = getattr(tokenizer, "unk_token_id", None)
    if any(i is None or i == unk for i in ids):
        raise ValueError("tokenizer has no <think>/</think> tokens; is this a Qwen3 tokenizer?")
    return int(ids[0]), int(ids[1])


def expected_prompt_tail(tokenizer: Any) -> list[int]:
    return list(tokenizer.encode(EMPTY_THINK_BLOCK, add_special_tokens=False))


def check_non_thinking(
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]] | None,
    response_ids: Sequence[Sequence[int]],
    *,
    where: str = "",
) -> dict[str, float]:
    """Return counts and raise :class:`ThinkingModeError` if any response carries think tags.

    ``prompt_ids`` may be None when prompts are unavailable; then only responses are checked.
    Prompts are checked for the empty think block at their tail (a WARNING count, not an error,
    because a custom system prompt could legitimately change the template output).
    """
    think_id, end_id = think_token_ids(tokenizer)
    n = len(response_ids)
    bad_resp = sum(1 for ids in response_ids if think_id in ids or end_id in ids)
    bad_prompt = 0
    if prompt_ids is not None:
        tail = expected_prompt_tail(tokenizer)
        for ids in prompt_ids:
            ids = [int(i) for i in ids]
            if ids[-len(tail) :] != tail:
                bad_prompt += 1
    stats = {
        "mixed_cuts/frac_responses_with_think_tag": (bad_resp / n) if n else 0.0,
        "mixed_cuts/frac_prompts_without_empty_think_block": (bad_prompt / n)
        if (n and prompt_ids is not None)
        else 0.0,
    }
    if bad_resp:
        raise ThinkingModeError(
            f"Qwen3 thinking mode detected{(' ' + where) if where else ''}: {bad_resp}/{n} responses contain "
            f"<think>/</think>; {bad_prompt}/{n} prompts lack the empty <think></think> block. "
            "Set data.apply_chat_template_kwargs.enable_thinking=false (training) / "
            "chat_template_kwargs.enable_thinking=false (eval)."
        )
    return stats
