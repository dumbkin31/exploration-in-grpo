"""Final-answer extraction: the last ``\\boxed{...}`` in a response (brace-balanced).

Same algorithm as verl's ``verl.utils.reward_score.math_dapo.last_boxed_only_string`` but
returning the *content* and tolerating an unterminated box at the end of a truncated response
(returns ``None`` in that case). A plain-text ``Answer: ...`` last line is offered as a fallback
for DAPO-style outputs.
"""

from __future__ import annotations

import re

_BOXED = "\\boxed{"
_ANSWER_LINE = re.compile(r"^\s*(?:\*\*)?Answer(?:\*\*)?\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def last_boxed_content(text: str) -> str | None:
    """Content of the last ``\\boxed{...}``; ``None`` if absent or unbalanced."""
    if not text:
        return None
    start = text.rfind(_BOXED)
    if start < 0:
        return None
    depth = 0
    i = start + len(_BOXED) - 1  # index of the opening brace
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(_BOXED) : i]
        i += 1
    return None


def last_answer_line(text: str) -> str | None:
    """The value after the last ``Answer:`` line, for responses that never box the answer."""
    matches = _ANSWER_LINE.findall(text or "")
    return matches[-1].strip() if matches else None


def normalize_letter(answer: str | None) -> str | None:
    """Map ``"(B)"``, ``"b"``, ``"\\text{B}"`` ... to ``"B"`` for multiple-choice benchmarks."""
    if answer is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    s = re.sub(r"[^A-Za-z]", "", s).upper()
    return s if len(s) == 1 else None
