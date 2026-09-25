from __future__ import annotations

import pytest

from mixed_cuts.thinking import EMPTY_THINK_BLOCK, ThinkingModeError, check_non_thinking, expected_prompt_tail


class FakeTok:
    """Qwen3-like ids: <think>=151667, </think>=151668, '\\n\\n'=271."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, toks):
        return [{"<think>": 151667, "</think>": 151668}[t] for t in toks]

    def encode(self, text, add_special_tokens=False):
        assert text == EMPTY_THINK_BLOCK
        return [151667, 271, 151668, 271]


def test_non_thinking_prompts_and_clean_responses_pass():
    tok = FakeTok()
    tail = expected_prompt_tail(tok)
    prompts = [[1, 2, 3, *tail], [9, *tail]]
    responses = [[5, 6, 7], [8]]
    stats = check_non_thinking(tok, prompts, responses)
    assert stats == {
        "mixed_cuts/frac_responses_with_think_tag": 0.0,
        "mixed_cuts/frac_prompts_without_empty_think_block": 0.0,
    }
    stats = check_non_thinking(tok, [[1, 2, 3]], [[5]])  # prompt without the block: counted, not fatal
    assert stats["mixed_cuts/frac_prompts_without_empty_think_block"] == 1.0


def test_think_tag_in_a_response_fails_loudly():
    tok = FakeTok()
    with pytest.raises(ThinkingModeError, match="1/2 responses contain"):
        check_non_thinking(tok, None, [[151667, 4, 151668, 5], [6]], where="in eval generation")
