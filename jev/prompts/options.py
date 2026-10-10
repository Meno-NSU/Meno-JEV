"""Knobs that control how an item becomes a prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PromptOptions:
    system_prompt: Optional[str] = None

    # Appended after the chat generation prompt, right before the continuation.
    # Empty by default: a chat template already opens the assistant turn.
    answer_prefix: str = ""

    # Prefix the scored continuation with a space (helps word-piece models).
    continuation_lead_space: bool = True

    # Words used for the binary (noul) answer space.
    noul_words: Tuple[str, str] = ("true", "false")  # (true, false)

    # Show "A. option" labels when listing the answer space.
    label_options: bool = True

    # How to shrink an over-long state: "head", "tail", "head_tail", "none".
    state_fit: str = "head_tail"

    # Tokens kept free for the answer space / prefix.
    reserve_tokens: int = 32

    # Newline separator between state, instruction and options.
    sep: str = "\n\n"
