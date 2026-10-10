"""Aggregation of predictions into accuracy / ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional


def _acc(preds: List) -> float:
    if not preds:
        return 0.0
    return sum(1 for p in preds if p.correct) / len(preds)


def _group(preds: List, key):
    out: DefaultDict[Any, List] = defaultdict(list)
    for p in preds:
        out[key(p)].append(p)
    return out


def _bucket_metrics(preds: List) -> Dict[str, Any]:
    n = len(preds)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for p in preds if p.correct)
    probs = [p.prob_gold for p in preds]
    ranks = [p.rank_gold for p in preds if p.rank_gold]
    mrr = (
        sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0.0
    )
    errs = sum(1 for p in preds if p.error)
    return {
        "n": n,
        "accuracy": correct / n,
        "mean_prob_gold": sum(probs) / n,
        "mrr": mrr,
        "errors": errs,
    }


def summarize(preds: List, cfg: Optional[object] = None) -> Dict[str, Any]:
    overall = _bucket_metrics(preds)
    by_type = {k: _bucket_metrics(v) for k, v in _group(preds, lambda p: p.question_type).items()}
    by_source = {k: _bucket_metrics(v) for k, v in _group(preds, lambda p: p.source).items()}
    by_split = {k: _bucket_metrics(v) for k, v in _group(preds, lambda p: p.split).items()}

    per_type_n = {k: v["n"] for k, v in by_type.items()}
    macro = (
        sum(v["accuracy"] for v in by_type.values()) / len(by_type) if by_type else 0.0
    )

    summary = {
        "overall": overall,
        "macro_accuracy_by_type": macro,
        "by_question_type": by_type,
        "by_source": by_source,
        "by_split": by_split,
        "per_type_counts": per_type_n,
    }
    if cfg is not None:
        summary["config"] = {
            "adapter": getattr(cfg, "adapter", None),
            "model_id": getattr(cfg, "model_id", None),
            "score_mode": getattr(cfg, "score_mode", None),
            "normalize": getattr(cfg, "normalize", None),
            "temperature": getattr(cfg, "temperature", None),
        }
    return summary
