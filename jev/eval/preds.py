"""Load saved predictions so metrics can be recomputed without re-running."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import List

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
