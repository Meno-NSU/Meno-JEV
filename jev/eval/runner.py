"""The evaluation loop: item -> prompt -> distribution -> prediction."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable

from ..data.loader import iter_benchmark
from ..data.loader import filter_items, sample_items
from ..models.base import ModelConfig, build_model
from ..prompts.options import PromptOptions
from ..scoring.modes import score
from ..tasks.base import resolve_recipe


@dataclass
class RunConfig:
    data_path: str = "benchmark.json"
    adapter: str = "hf"
    model_id: Optional[str] = None
    sources: Optional[Sequence[str]] = None
    question_types: Optional[Sequence[str]] = None
    splits: Optional[Sequence[str]] = None
    limit: Optional[int] = None
    per_source_limit: Optional[int] = None
    sample: Optional[int] = None           # small stratified subsample size
    seed: int = 0

    score_mode: Optional[str] = None       # override recipe's default
    normalize: str = "length"
    temperature: float = 1.0
    batch_size: Optional[int] = None       # candidates per call (None = adapter-managed)
    top_k: int = 5

    device: Optional[str] = None
    dtype: str = "auto"
    max_context_tokens: Optional[int] = None
    chat_template: Optional[str] = None
    attn_implementation: Optional[str] = None

    output_dir: str = "outputs"
    run_name: Optional[str] = None
    prompt: PromptOptions = field(default_factory=PromptOptions)
    adapter_extra: Dict[str, Any] = field(default_factory=dict)

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            extra=dict(self.adapter_extra),
            model_id=self.model_id,
            device=self.device,
            dtype=self.dtype,
            max_context_tokens=self.max_context_tokens,
            chat_template=self.chat_template,
            attn_implementation=self.attn_implementation,
            system_prompt=self.prompt.system_prompt,
        )


@dataclass
class Prediction:
    id: str
    source: str
    question_type: str
    split: str
    gold_id: str
    pred_id: str
    correct: bool
    prob_gold: float
    rank_gold: Optional[int]
    score_mode: str
    top: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _select_items(cfg: RunConfig):
    items = filter_items(
        iter_benchmark(cfg.data_path),
        sources=cfg.sources,
        question_types=cfg.question_types,
        splits=cfg.splits,
    )
    if cfg.per_source_limit:
        counts: Dict[str, int] = {}
        kept = []
        for it in items:
            n = counts.get(it.source, 0)
            if n < cfg.per_source_limit:
                kept.append(it)
                counts[it.source] = n + 1
            items = kept
    if cfg.sample:
        items = sample_items(items, cfg.sample, cfg.seed)
    if cfg.limit:
        items = items[: cfg.limit]
    return items


def _top(dist, candidates, k: int) -> List[Dict[str, Any]]:
    order = sorted(range(len(dist.ids)), key=lambda i: dist.probs[i], reverse=True)[:k]
    text_by_id = {c.id: c.text for c in candidates}
    return [
        {"id": dist.ids[i], "prob": dist.probs[i], "text": text_by_id.get(dist.ids[i], "")}
        for i in order
    ]


def run(cfg: RunConfig, adapter=None) -> Dict[str, Any]:
    """Run the benchmark and return a summary dict (also written to disk)."""
    from ..eval.metrics import summarize

    items = _select_items(cfg)
    if adapter is None:
        adapter = build_model(cfg.adapter, cfg.model_config(), load=True)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
    pred_path = out_dir / f"{run_name}.predictions.jsonl"

    preds: List[Prediction] = []
    with open(pred_path, "w", encoding="utf-8") as fh:
        for item in tqdm(items, desc="eval", unit="item"):
            try:
                recipe = resolve_recipe(item)
                rq = recipe.build(item, adapter, cfg.prompt, cfg.score_mode)
                dist = score(
                    adapter,
                    rq.context,
                    rq.candidates,
                    mode=rq.score_mode,
                    normalize=cfg.normalize,
                    temperature=cfg.temperature,
                    batch_size=cfg.batch_size,
                )
                pred_id = dist.argmax_id()
                if item.gold_id not in dist.ids:
                    # gold option absent (e.g. truncated) -> count as wrong
                    pass
                pred = Prediction(
                    id=item.id,
                    source=item.source,
                    question_type=item.question_type,
                    split=item.split,
                    gold_id=item.gold_id,
                    pred_id=pred_id,
                    correct=(pred_id == item.gold_id),
                    prob_gold=dist.prob_of(item.gold_id),
                    rank_gold=dist.rank_of(item.gold_id),
                    score_mode=rq.score_mode,
                    top=_top(dist, rq.candidates, cfg.top_k),
                )
            except Exception as exc:  # keep going on per-item failures
                pred = Prediction(
                    id=item.id,
                    source=item.source,
                    question_type=item.question_type,
                    split=item.split,
                    gold_id=item.gold_id,
                    pred_id="",
                    correct=False,
                    prob_gold=0.0,
                    rank_gold=None,
                    score_mode=cfg.score_mode or "",
                    error=f"{type(exc).__name__}: {exc}",
                )
            preds.append(pred)
            fh.write(pred.to_json() + "\n")
            fh.flush()

    summary = summarize(preds, cfg=cfg)
    summary["predictions_path"] = str(pred_path)
    summary["run_name"] = run_name
    return summary
