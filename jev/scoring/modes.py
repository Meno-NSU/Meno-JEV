"""Two ways to turn an LLM into a scorer.

``continuation``
    Append each candidate's text after the answer prefix and sum the
    log-probabilities of its tokens. Optionally length-normalise. Robust for
    free-text options of any length.

``next_token``
    Put the answer prefix last and read the log-probabilities of the *first*
    token of each candidate (a letter, a digit, "true"/"false"). Cheap and
    precise, but only valid when candidates are distinguishable by their first
    token.

Both return a :class:`~jev.scoring.distribution.Distribution` over candidate ids.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..data.schema import Candidate
from ..models.base import ModelAdapter
from .distribution import Distribution, softmax

SCORE_MODES = ("continuation", "next_token")

NORMALIZATIONS = ("length", "sum", "none")


def _normalize(value: float, num_tokens: int, how: str) -> float:
    if how == "length":
        return value / max(num_tokens, 1)
    return value


def score_continuation(
    adapter: ModelAdapter,
    context: str,
    candidates: Sequence[Candidate],
    normalize: str = "length",
    temperature: float = 1.0,
    batch_size: Optional[int] = None,
) -> Distribution:
    """Score every candidate as a continuation of ``context``.

    ``batch_size`` caps how many candidates go into one ``loglikelihood_batch``
    call. ``None`` (default) passes all candidates at once and lets the model
    adapter do memory-aware batching internally (recommended: the adapter knows
    the token budget and can retry on OOM).
    """
    ids = [c.id for c in candidates]
    if not candidates:
        return Distribution([], [], [])

    pairs = [(context, c.continuation or c.text) for c in candidates]
    scores: List[float] = []
    meta_scores: List[float] = []

    if batch_size and batch_size > 0:
        chunks = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
    else:
        chunks = [pairs]
    for chunk in chunks:
        for cs in adapter.loglikelihood_batch(chunk):
            scores.append(_normalize(cs.logprob, cs.num_tokens, normalize))
            meta_scores.append(cs.logprob)

    probs = softmax(scores, temperature)
    return Distribution(ids, scores, probs, meta={"raw_logprobs": meta_scores})


def score_next_token(
    adapter: ModelAdapter,
    context: str,
    candidates: Sequence[Candidate],
    temperature: float = 1.0,
    token_field: str = "token",
) -> Distribution:
    if not candidates:
        return Distribution([], [], [])
    ids = [c.id for c in candidates]
    tokens = [getattr(c, token_field) or c.text for c in candidates]
    token_ids = [adapter.encode(t, add_special_tokens=False) for t in tokens]
    logprobs = adapter.next_token_logprobs(context, token_ids)
    probs = softmax(list(logprobs), temperature)
    return Distribution(ids, list(logprobs), probs, meta={"tokens": tokens})


def score(
    adapter: ModelAdapter,
    context: str,
    candidates: Sequence[Candidate],
    mode: str = "continuation",
    normalize: str = "length",
    temperature: float = 1.0,
    batch_size: Optional[int] = None,
) -> Distribution:
    if mode == "next_token":
        return score_next_token(adapter, context, candidates, temperature)
    if mode != "continuation":
        raise ValueError(f"unknown score mode '{mode}'. expected one of {SCORE_MODES}")
    return score_continuation(
        adapter, context, candidates, normalize, temperature, batch_size
    )
