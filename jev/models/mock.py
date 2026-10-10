"""A dependency-free mock adapter.

It implements the :class:`ModelAdapter` interface with no torch/transformers so
the full pipeline (tasks, scoring, metrics, CLI) can be exercised on a CPU-only
machine and inside the test suite.

Scoring is fully controllable: pass ``scores_fn(context, continuation) -> float``
in ``ModelConfig.extra`` to decide log-probabilities, or set
``preferred_substrings`` to a list of strings; a continuation containing one of
them scores high. With no configuration every candidate ties (uniform
distribution).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .base import ContinuationScore, ModelAdapter, ModelConfig, register_model


def _simple_tokens(text: str) -> List[str]:
    # A whitespace/punct tokenizer is enough for a mock.
    out: List[str] = []
    buf = ""
    for ch in text:
        if ch.isspace() or ch in ".,:;!?()[]{}\"'":
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return [t for t in out if t.strip()]


@register_model("mock")
class MockAdapter(ModelAdapter):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__(config)
        self._scores_fn: Optional[Callable[[str, str], float]] = self.config.extra.get(
            "scores_fn"
        )
        self._preferred: List[str] = list(self.config.extra.get("preferred_substrings", []))
        self._vocab: Dict[str, int] = {}
        self._template = "mock-context"

    def load(self) -> None:
        return None

    @property
    def max_context_tokens(self) -> int:
        return int(self.config.max_context_tokens or 4096)

    def _id(self, tok: str) -> int:
        return self._vocab.setdefault(tok, len(self._vocab) + 1)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [self._id(t) for t in _simple_tokens(text)]

    def decode(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)

    def render_chat(
        self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        parts = [f"<{m.get('role', 'user')}>{m.get('content', '')}" for m in messages]
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n<assistant>"
        return text

    def loglikelihood(self, context: str, continuation: str) -> ContinuationScore:
        tok = _simple_tokens(continuation)
        if self._scores_fn is not None:
            lp = float(self._scores_fn(context, continuation))
        elif self._preferred:
            lp = 5.0 if any(p in continuation for p in self._preferred) else -5.0
        else:
            lp = 0.0
        return ContinuationScore(lp * max(len(tok), 1), max(len(tok), 1), True)

    def next_token_logprobs(
        self, context: str, token_ids: Sequence[Sequence[int]]
    ) -> List[float]:
        inv = {v: k for k, v in self._vocab.items()}
        out: List[float] = []
        for ids in token_ids:
            tok = inv.get(ids[0], "") if ids else ""
            if self._scores_fn is not None:
                out.append(float(self._scores_fn(context, tok)))
            elif self._preferred:
                out.append(2.0 if tok in self._preferred else -2.0)
            else:
                out.append(0.0)
        return out
