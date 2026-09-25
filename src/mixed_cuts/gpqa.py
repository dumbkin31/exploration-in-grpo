"""Multiple-choice answer extraction for GPQA (letters A-D), separate from the math path.

GPQA answers are letters, not boxed numerics, so the digit rule of the math reward would score it
at zero for parsing reasons alone. This parser tries, in order:

1. the content of the FINAL ``\\boxed{...}`` when it names a single option: ``\\boxed{C}``,
   ``\\boxed{(C)}``, ``\\boxed{\\text{C}}``, ``\\boxed{C) the option text}``;
2. an explicit statement: ``the answer is (C)``, ``Answer: C``, ``**Answer**: (C)``,
   ``correct option is C``, ``choice: C``, ``option C``;
3. the last standalone letter A-D in the text (``(C)``, ``C.``, ``C)``), ignoring the English
   article "A" (an "A" followed by a lowercase word).

Returns ``None`` when nothing matches. Everything is case-insensitive; the result is upper case.
"""

from __future__ import annotations

import re

from mixed_cuts.boxed import last_boxed_content

LETTERS = "ABCD"

_TEXT_CMD = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{([^}]*)\}")
_BOXED_LEADING_LETTER = re.compile(r"^\s*\(?\s*([A-Da-d])\s*(?:[).:\-]|$)")
_EXPLICIT = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:|=)?\s*(?:\*\*)?\s*\(?\s*([A-Da-d])\s*\)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_STANDALONE = re.compile(r"(?<![A-Za-z0-9\\])\(?([A-Da-d])\)?(?![A-Za-z0-9])")


def _letter_from_boxed(content: str) -> str | None:
    s = _TEXT_CMD.sub(r"\1", content)
    s = s.replace("*", "").strip()
    stripped = re.sub(r"[^A-Za-z]", "", s)
    if len(stripped) == 1 and stripped.upper() in LETTERS:
        return stripped.upper()
    m = _BOXED_LEADING_LETTER.match(s)
    return m.group(1).upper() if m else None


def extract_choice_letter(text: str) -> str | None:
    """Return "A"/"B"/"C"/"D" or None. See the module docstring for the fallback order."""
    if not text:
        return None
    boxed = last_boxed_content(text)
    if boxed is not None:
        letter = _letter_from_boxed(boxed)
        if letter:
            return letter
    explicit = _EXPLICIT.findall(text)
    if explicit:
        return explicit[-1].upper()
    last: str | None = None
    for m in _STANDALONE.finditer(text):
        letter = m.group(1)
        after = text[m.end() : m.end() + 2]
        # the English article "A" (e.g. "A triangle") is not an answer
        if letter.upper() == "A" and m.group(0) == "A" and re.match(r"\s[a-z]", after):
            continue
        last = letter.upper()
    return last
