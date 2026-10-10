"""A probability distribution over candidate answers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    temp = temperature if temperature and temperature > 0 else 1.0
    scaled = [s / temp for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


@dataclass
class Distribution:
    """Scores and probabilities for a set of candidate ids.

    ``scores`` are the (possibly normalised) values fed into the softmax;
    ``probs`` are the resulting probabilities. Both are aligned with ``ids``.
    """

    ids: List[str]
    scores: List[float]
    probs: List[float]
    meta: Dict[str, object] = field(default_factory=dict)

    def argmax_id(self) -> str:
        best_i = max(range(len(self.ids)), key=lambda i: self.probs[i])
        return self.ids[best_i]

    def prob_of(self, cid: str) -> float:
        try:
            return self.probs[self.ids.index(cid)]
        except ValueError:
            return 0.0

    def score_of(self, cid: str) -> float:
        try:
            return self.scores[self.ids.index(cid)]
        except ValueError:
            return float("nan")

    def rank_of(self, cid: str, ascending: bool = False) -> Optional[int]:
        """1-based rank of ``cid`` by probability (higher prob -> rank 1)."""
        if cid not in self.ids:
            return None
        order = sorted(range(len(self.ids)), key=lambda i: self.probs[i], reverse=not ascending)
        for r, i in enumerate(order, start=1):
            if self.ids[i] == cid:
                return r
        return None

    def as_dict(self) -> Dict[str, float]:
        return {cid: p for cid, p in zip(self.ids, self.probs)}
