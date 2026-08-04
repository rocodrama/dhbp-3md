from __future__ import annotations

import re


def strip_gemma_thought_channel(text: str) -> str:
    """Return visible answer text when Gemma-style thought tokens leak into content."""
    for marker in ("<unused95>", "<|channel|>"):
        if marker in text:
            return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def has_answer_line(text: str) -> bool:
    return re.search(r"(?m)^ANSWER: ", text) is not None

