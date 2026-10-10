"""Loading and filtering of the benchmark file.

The benchmark is a single large JSON array (~113 MB). We load it with the
standard library; on machines with little RAM an optional streaming path using
``ijson`` is provided, but it is not required.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

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


def _round_robin_sample(items: List[Item], n: int, rng: random.Random) -> List[Item]:
    """Take up to ``n`` items, cycling across sources so they stay diverse."""
    by_source: Dict[str, List[Item]] = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)
    for bucket in by_source.values():
        rng.shuffle(bucket)
    sources = list(by_source)
    rng.shuffle(sources)

    out: List[Item] = []
    idx: Dict[str, int] = {s: 0 for s in sources}
    while len(out) < n and sources:
        progressed = False
        for s in list(sources):
            if idx[s] < len(by_source[s]):
                out.append(by_source[s][idx[s]])
                idx[s] += 1
                progressed = True
                if len(out) >= n:
                    break
            else:
                sources.remove(s)
        if not progressed:
            break
    return out


def sample_items(
    items: Sequence[Item],
    n: int,
    seed: int = 0,
    stratify_by: str = "question_type",
) -> List[Item]:
    """Draw a small, reproducible, stratified subsample of ``items``.

    The per-question-type counts are allocated proportionally to the full set
    (so ``choice``/``noul``/``score`` keep roughly their real mix), and within
    each type items are picked round-robin across sources for diversity. This is
    meant for quick prompt tuning without running the whole benchmark.
    """
    items = list(items)
    if n <= 0 or n >= len(items):
        return items

    groups: Dict[str, List[Item]] = defaultdict(list)
    for it in items:
        key = getattr(it, stratify_by) or "all"
        groups[key].append(it)

    rng = random.Random(seed)
    total = len(items)
    # Largest-remainder allocation so the pieces sum to exactly n.
    exact = {k: n * len(v) / total for k, v in groups.items()}
    alloc = {k: min(int(val), len(groups[k])) for k, val in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True):
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < len(groups[k]):
            alloc[k] += 1
    # Make sure every present type is represented at least once.
    for k, v in groups.items():
        if alloc.get(k, 0) == 0 and sum(alloc.values()) < n:
            alloc[k] = 1

    out: List[Item] = []
    for k, bucket in groups.items():
        out.extend(_round_robin_sample(bucket, alloc.get(k, 0), rng))
    rng.shuffle(out)
    return out[:n]
