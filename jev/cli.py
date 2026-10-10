"""Command line entry point: ``menoj <command> [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _read_maybe_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return value


def cmd_models(_args) -> int:
    from .models.base import available_models
    from .models import _registry  # noqa: F401

    print("Available model adapters:")
    for name in available_models():
        print(f"  - {name}")
    return 0


def cmd_sources(args) -> int:
    from .data.loader import iter_benchmark
    from collections import Counter, defaultdict

    counts = Counter()
    types = defaultdict(set)
    for item in iter_benchmark(args.data):
        counts[item.source] += 1
        types[item.source].add(item.question_type)
    print(f"{'source':30s} {'n':>5s}  types")
    for src in sorted(counts):
        print(f"{src:30s} {counts[src]:5d}  {','.join(sorted(types[src]))}")
    return 0


def cmd_inspect(args) -> int:
    from .data.loader import iter_benchmark, filter_items

    items = filter_items(
        iter_benchmark(args.data),
        sources=_csv(args.source),
        ids=[args.id] if args.id else None,
    )
    if args.id:
        items = [it for it in items if it.id == args.id]
    for item in items[: args.limit or 3]:
        print("=" * 78)
        print(f"id={item.id}  source={item.source}  type={item.question_type}  split={item.split}")
        print(f"instruction: {item.instructions}")
        print(f"gold_id: {item.gold_id!r}  ({len(item.candidates)} candidates)")
        for c in item.candidates[:8]:
            mark = "*" if c.id == item.gold_id else " "
            print(f"  {mark} id={c.id!r} text={c.text!r} token={c.token!r}")
        if len(item.candidates) > 8:
            print(f"  ... {len(item.candidates) - 8} more")
    return 0


def cmd_smoke(args) -> int:
    from .eval.smoke import SmokeConfig, run_smoke, render_smoke

    modes = tuple(m.strip() for m in (args.modes or "continuation,next_token").split(",") if m.strip())
    cfg = SmokeConfig(
        data_path=args.data,
        per_source_limit=args.per_source_limit,
        modes=modes,
        sample=args.sample,
        seed=args.seed,
    )
    summary = run_smoke(cfg)
    print(render_smoke(summary))
    return 0 if summary["ok"] else 1


def _build_run_config(args):
    from .eval.runner import RunConfig
    from .prompts.options import PromptOptions

    prompt = PromptOptions(
        system_prompt=args.system_prompt,
        answer_prefix=args.answer_prefix or "",
        state_fit=args.state_fit,
    )
    return RunConfig(
        data_path=args.data,
        adapter=args.adapter,
        model_id=args.model_id,
        sources=_csv(args.sources),
        question_types=_csv(args.types),
        splits=_csv(args.split),
        limit=args.limit,
        per_source_limit=args.per_source_limit,
        score_mode=args.score_mode,
        normalize=args.normalize,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        dtype=args.dtype,
        max_context_tokens=args.max_context_tokens,
        chat_template=_read_maybe_path(args.chat_template),
        attn_implementation=args.attn,
        output_dir=args.output_dir,
        run_name=args.run_name,
        prompt=prompt,
    )


def cmd_run(args) -> int:
    from .eval.runner import run
    from .eval.report import write_json, write_markdown, render_markdown

    cfg = _build_run_config(args)
    summary = run(cfg)

    out_dir = Path(cfg.output_dir)
    name = summary.get("run_name")
    write_json(summary, out_dir / f"{name}.report.json")
    write_markdown(summary, out_dir / f"{name}.report.md")
    print("\n" + render_markdown(summary))
    print(f"predictions: {summary.get('predictions_path')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="menoj", description="Loglikelihood JEV evaluator")
    sub = p.add_subparsers(dest="command", required=True)

    pm = sub.add_parser("models", help="list available model adapters")
    pm.set_defaults(func=cmd_models)

    ps = sub.add_parser("sources", help="list sources in the dataset")
    ps.add_argument("--data", default="benchmark.json")
    ps.set_defaults(func=cmd_sources)

    pi = sub.add_parser("inspect", help="show parsed items")
    pi.add_argument("--data", default="benchmark.json")
    pi.add_argument("--source")
    pi.add_argument("--id")
    pi.add_argument("--limit", type=int, default=3)
    pi.set_defaults(func=cmd_inspect)

    pk = sub.add_parser("smoke", help="self-check the harness on a small subsample")
    pk.add_argument("--data", default="benchmark.json")
    pk.add_argument("--per-source-limit", type=int, default=3)
    pk.add_argument("--modes", default="continuation,next_token")
    pk.add_argument("--sample", choices=["head", "random"], default="head")
    pk.add_argument("--seed", type=int, default=0)
    pk.set_defaults(func=cmd_smoke)

    pr = sub.add_parser("run", help="run the evaluation")
    pr.add_argument("--data", default="benchmark.json")
    pr.add_argument("--adapter", default="hf", help="model adapter name (hf, qwen, mock, ...)")
    pr.add_argument("--model-id", default=None, help="local path or HF hub id")
    pr.add_argument("--sources", default=None, help="comma-separated source filter")
    pr.add_argument("--types", default=None, help="comma-separated question types")
    pr.add_argument("--split", default=None, help="comma-separated splits (train,dev,test)")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--per-source-limit", type=int, default=None)
    pr.add_argument("--score-mode", choices=["continuation", "next_token"], default=None)
    pr.add_argument("--normalize", choices=["length", "sum", "none"], default="length")
    pr.add_argument("--temperature", type=float, default=1.0)
    pr.add_argument("--top-k", type=int, default=5)
    pr.add_argument("--device", default=None)
    pr.add_argument("--dtype", default="auto")
    pr.add_argument("--max-context-tokens", type=int, default=None)
    pr.add_argument("--chat-template", default=None, help="path to a template file or literal")
    pr.add_argument("--attn", default=None, help="attn_implementation (e.g. sdpa, flash_attention_2)")
    pr.add_argument("--system-prompt", default=None)
    pr.add_argument("--answer-prefix", default=None)
    pr.add_argument("--state-fit", choices=["head", "tail", "head_tail", "none"], default="head_tail")
    pr.add_argument("--output-dir", default="outputs")
    pr.add_argument("--run-name", default=None)
    pr.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
