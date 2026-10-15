"""Load saved predictions so metrics can be recomputed without re-running."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional

from .runner import Prediction

_FIELDS = {f.name for f in fields(Prediction)}


def load_predictions(path) -> List[Prediction]:
    """Read a ``*.predictions.jsonl`` file into :class:`Prediction` objects.

    Only the stored fields are used; unknown keys are ignored and missing ones
    fall back to the dataclass defaults (so older files still load).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"predictions not found: {p}")
    preds: List[Prediction] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            preds.append(Prediction(**{k: v for k, v in data.items() if k in _FIELDS}))
    return preds


def failed_ids(path) -> List[str]:
    """Item ids that failed (have an ``error``) in a predictions file."""
    return [p.id for p in load_predictions(path) if getattr(p, "error", None)]


def write_predictions(preds: List[Prediction], path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for pred in preds:
            fh.write(json.dumps(asdict(pred), ensure_ascii=False) + "\n")
    return p


def merge_predictions(base_path, retry_path, out_path=None) -> List[Prediction]:
    """Overlay ``retry`` predictions onto ``base`` (keyed by item id).

    Retried rows replace the originals in place; ids only present in ``retry``
    are appended. Call repeatedly (retry a merged file) to chase residuals.
    """
    base = load_predictions(base_path)
    retry = load_predictions(retry_path)
    retry_by_id: Dict[str, Prediction] = {p.id: p for p in retry}

    merged: List[Prediction] = []
    seen = set()
    for p in base:
        merged.append(retry_by_id.get(p.id, p))
        seen.add(p.id)
    for p in retry:
        if p.id not in seen:
            merged.append(p)
    if out_path is not None:
        write_predictions(merged, out_path)
    return merged
