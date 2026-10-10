"""A self-check for the harness: does the benchmark run end-to-end correctly?

This does **not** measure model quality. It uses the dependency-free ``mock``
adapter to verify that, on a small subsample of every source, the whole pipeline
(parse -> recipe -> render -> score -> distribution) is structurally sound:

* every source parses and resolves a recipe;
* every rendered query has candidates and a gold candidate that exists in them;
* candidate counts match the raw criteria;
* the distribution is a valid probability vector over exactly the candidates;
* no item raises an error, in both scoring modes.

Run with ``python -m jev.cli smoke`` (or ``run_smoke`` from Python). Exit code
is non-zero if any check fails, so it is safe to use in CI / on the cluster.
"""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.loader import iter_benchmark
from ..data.schema import CHOICE, NOUL, SCORE, Item
from ..models.base import ModelConfig, build_model
from ..prompts.options import PromptOptions
from ..scoring.modes import score
from ..tasks.base import resolve_recipe


@dataclass
class SmokeConfig:
    data_path: str = "benchmark.json"
    per_source_limit: int = 3
    modes: Sequence[str] = ("continuation", "next_token")
    sample: str = "head"          # "head" (deterministic) or "random"
    seed: int = 0
    prompt: PromptOptions = field(default_factory=PromptOptions)


@dataclass
class ItemCheck:
    item_id: str
    source: str
    question_type: str
    mode: str
    ok: bool
    issues: List[str] = field(default_factory=list)


def _expected_candidate_count(item: Item) -> Optional[int]:
    if item.question_type == CHOICE and isinstance(item.criteria, dict):
        return len(item.criteria)
    if item.question_type == NOUL:
        return 2
    if item.question_type == SCORE and isinstance(item.criteria, list):
        return len(item.criteria)
    return None


def check_item(item: Item, adapter, mode: str, options: PromptOptions) -> ItemCheck:
    issues: List[str] = []
    try:
        recipe = resolve_recipe(item)
        rq = recipe.build(item, adapter, options, mode)
    except Exception as exc:  # noqa: BLE001
        return ItemCheck(item.id, item.source, item.question_type, mode, False,
                         [f"build failed: {type(exc).__name__}: {exc}"])

    if not rq.candidates:
        issues.append("no candidates")
    if not rq.context or not rq.context.strip():
        issues.append("empty context")

    cand_ids = [c.id for c in rq.candidates]
    if len(set(cand_ids)) != len(cand_ids):
        issues.append("duplicate candidate ids")
    if item.gold_id not in cand_ids:
        issues.append(f"gold id {item.gold_id!r} not among candidates")

    expected = _expected_candidate_count(item)
    if expected is not None and len(rq.candidates) != expected:
        issues.append(f"candidate count {len(rq.candidates)} != criteria {expected}")

    try:
        dist = score(
            adapter, rq.context, rq.candidates,
            mode=rq.score_mode, normalize=rq.normalize, temperature=rq.temperature,
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"score failed: {type(exc).__name__}: {exc}")
        return ItemCheck(item.id, item.source, item.question_type, mode, False, issues)

    if dist.ids != cand_ids:
        issues.append("distribution ids != candidate ids")
    if len(dist.probs) != len(cand_ids):
        issues.append("probability vector length mismatch")
    if not all(math.isfinite(p) and p >= 0.0 for p in dist.probs):
        issues.append("non-finite or negative probability")
    total = sum(dist.probs)
    if dist.probs and abs(total - 1.0) > 1e-4:
        issues.append(f"probabilities sum to {total:.6f}")
    if dist.ids and dist.argmax_id() not in cand_ids:
        issues.append("argmax id not a candidate")

    return ItemCheck(item.id, item.source, item.question_type, mode, not issues, issues)


def run_smoke(cfg: Optional[SmokeConfig] = None) -> Dict[str, Any]:
    cfg = cfg or SmokeConfig()
    adapter = build_model("mock", ModelConfig(), load=True)

    by_source: "OrderedDict[str, List[Item]]" = OrderedDict()
    for item in iter_benchmark(cfg.data_path):
        by_source.setdefault(item.source, []).append(item)

    rng = random.Random(cfg.seed)
    checks: List[ItemCheck] = []
    for source, items in by_source.items():
        if cfg.sample == "random":
            sample = rng.sample(items, min(cfg.per_source_limit, len(items)))
        else:
            sample = items[: cfg.per_source_limit]
        for item in sample:
            for mode in cfg.modes:
                checks.append(check_item(item, adapter, mode, cfg.prompt))

    failures = [c for c in checks if not c.ok]
    per_source = defaultdict(lambda: {"n": 0, "fail": 0})
    for c in checks:
        per_source[c.source]["n"] += 1
        if not c.ok:
            per_source[c.source]["fail"] += 1

    type_counts = Counter()
    for source, items in by_source.items():
        for item in items[: cfg.per_source_limit]:
            type_counts[item.question_type] += 1

    summary = {
        "data_path": str(cfg.data_path),
        "sources": len(by_source),
        "items_loaded": sum(len(v) for v in by_source.values()),
        "checks_run": len(checks),
        "checks_failed": len(failures),
        "modes": list(cfg.modes),
        "per_source_limit": cfg.per_source_limit,
        "question_types_seen": dict(type_counts),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "failures": [
            {"id": f.item_id, "source": f.source, "type": f.question_type,
             "mode": f.mode, "issues": f.issues}
            for f in failures
        ],
        "ok": len(failures) == 0,
    }
    return summary


def render_smoke(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append("Meno-JEV smoke check")
    lines.append(
        f"  data: {summary['data_path']}  sources: {summary['sources']}  "
        f"items: {summary['items_loaded']}"
    )
    lines.append(
        f"  checks: {summary['checks_run']}  failed: {summary['checks_failed']}  "
        f"modes: {','.join(summary['modes'])}"
    )
    lines.append(f"  question types seen: {summary['question_types_seen']}")
    bad = {k: v for k, v in summary["per_source"].items() if v["fail"]}
    if bad:
        lines.append("  failing sources:")
        for k, v in sorted(bad.items()):
            lines.append(f"    - {k}: {v['fail']}/{v['n']} failed")
    for f in summary["failures"][:20]:
        lines.append(f"    ! {f['id']} [{f['mode']}]: {'; '.join(f['issues'])}")
    if len(summary["failures"]) > 20:
        lines.append(f"    ... {len(summary['failures']) - 20} more")
    lines.append("  RESULT: " + ("OK" if summary["ok"] else "FAILED"))
    return "\n".join(lines) + "\n"
