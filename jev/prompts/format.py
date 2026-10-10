"""Rendering raw benchmark fields into prompt text."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from ..data.schema import Candidate
from ..models.base import ModelAdapter
from .options import PromptOptions

_JSON_KEYS_OF_INTEREST = ("prompt", "question", "instruction", "query", "text", "input")


def render_state(state: str) -> str:
    """Render the raw ``state`` field.

    If it is a JSON object we surface the most useful field(s) instead of the
    raw blob; otherwise the text is returned unchanged.
    """
    s = (state or "").strip()
    if not s or s[0] not in "{[":
        return s
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return s
    if isinstance(obj, dict):
        for k in _JSON_KEYS_OF_INTEREST:
            if k in obj and isinstance(obj[k], str):
                rest = {kk: vv for kk, vv in obj.items() if kk != k}
                extra = ""
                if rest:
                    extra = "\n" + json.dumps(rest, ensure_ascii=False, indent=2)
                return obj[k] + extra
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _tokens(adapter: ModelAdapter, text: str) -> List[int]:
    return adapter.encode(text, add_special_tokens=False)


def fit_text(
    adapter: ModelAdapter, text: str, max_tokens: int, mode: str = "head_tail"
) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens."""
    if mode == "none" or max_tokens <= 0:
        return text
    ids = _tokens(adapter, text)
    if len(ids) <= max_tokens:
        return text
    if mode == "tail":
        return adapter.decode(ids[-max_tokens:])
    if mode == "head":
        return adapter.decode(ids[:max_tokens])
    # head_tail: keep both ends, join with a marker.
    head = max_tokens // 2
    tail = max_tokens - head
    return adapter.decode(ids[:head]) + "\n[...]\n" + adapter.decode(ids[-tail:])


def format_options(
    candidates: Sequence[Candidate], labeled: bool = True
) -> str:
    lines: List[str] = []
    for i, c in enumerate(candidates):
        if labeled:
            lines.append(f"{chr(ord('A') + i) if i < 26 else c.token}. {c.text}")
        else:
            lines.append(c.text)
    return "\n".join(lines)


def build_context(
    adapter: ModelAdapter,
    options: PromptOptions,
    state_text: str,
    instruction: str,
    options_block: Optional[str] = None,
) -> str:
    """Assemble the full prompt string up to (but excluding) the continuation."""
    parts: List[str] = []
    if state_text:
        parts.append(state_text)
    if instruction:
        parts.append(instruction)
    if options_block:
        parts.append(options_block)
    body = options.sep.join(parts)

    messages: List[Dict[str, str]] = []
    if options.system_prompt:
        messages.append({"role": "system", "content": options.system_prompt})
    messages.append({"role": "user", "content": body})

    rendered = adapter.render_chat(messages, add_generation_prompt=True)
    return rendered + options.answer_prefix


def state_budget(
    adapter: ModelAdapter,
    options: PromptOptions,
    instruction: str,
    options_block: Optional[str] = None,
) -> int:
    """How many tokens the state may occupy without overflowing context."""
    fixed = adapter.encode(instruction or "", add_special_tokens=False)
    fixed += adapter.encode(options_block or "", add_special_tokens=False)
    fixed += adapter.encode(options.answer_prefix or "", add_special_tokens=False)
    fixed += adapter.encode(
        "\n".join([options.system_prompt or "", "user", "assistant"]),
        add_special_tokens=False,
    )
    return max(adapter.max_context_tokens - len(fixed) - options.reserve_tokens, 64)
