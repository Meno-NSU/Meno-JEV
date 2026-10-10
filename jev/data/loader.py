"""Loading and filtering of the benchmark file.

The benchmark is a single large JSON array (~113 MB). We load it with the
standard library; on machines with little RAM an optional streaming path using
``ijson`` is provided, but it is not required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from .schema import Item, parse_item


def _coerce_path(path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "benchmark.json"
    return p


def iter_benchmark(path) -> Iterator[Item]:
    """Yield parsed :class:`Item` objects one at a time."""
    p = _coerce_path(path)
    if not p.exists():
        raise FileNotFoundError(f"benchmark not found: {p}")

    # Prefer ijson streaming if available (keeps memory low on big files).
    try:
        import ijson  # type: ignore

        with open(p, "rb") as fh:
            for record in ijson.items(fh, "item"):
                yield parse_item(record)
        return
    except ImportError:
        pass

    with open(p, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    for record in records:
        yield parse_item(record)


def load_benchmark(path) -> List[Item]:
    """Load the whole benchmark into a list of :class:`Item`."""
    return list(iter_benchmark(path))


def filter_items(
    items: Iterable[Item],
    sources: Optional[Sequence[str]] = None,
    question_types: Optional[Sequence[str]] = None,
    splits: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
    predicate: Optional[Callable[[Item], bool]] = None,
) -> List[Item]:
    """Filter items by source / question type / split / id / custom predicate."""
    src = set(sources) if sources else None
    qts = set(question_types) if question_types else None
    spl = set(splits) if splits else None
    idset = set(ids) if ids else None

    out: List[Item] = []
    for item in items:
        if src is not None and item.source not in src:
            continue
        if qts is not None and item.question_type not in qts:
            continue
        if spl is not None and item.split not in spl:
            continue
        if idset is not None and item.id not in idset:
            continue
        if predicate is not None and not predicate(item):
            continue
        out.append(item)
    return out
