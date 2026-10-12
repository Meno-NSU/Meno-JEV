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


def cmd_doctor(_args) -> int:
    import platform
    import sys

    print("Meno-JEV environment doctor")
    print(f"  python: {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"  executable: {sys.executable}")
    try:
        from . import __version__ as jev_version

        print(f"  menoj: {jev_version}")
    except Exception:  # noqa: BLE001
        pass

    def _try(name):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "?")
        except ModuleNotFoundError:
            return "NOT INSTALLED"
        except Exception as exc:  # noqa: BLE001
            return f"IMPORT ERROR ({type(exc).__name__}: {exc})"

    print(f"  torch: {_try('torch')}")
    print(f"  transformers: {_try('transformers')}")
    print(f"  accelerate: {_try('accelerate')}")
    print(f"  sentencepiece: {_try('sentencepiece')}")
    print(f"  tqdm: {_try('tqdm')}")

    # Probe the actual model import (top-level `transformers` can import fine
    # while `AutoModelForCausalLM` fails, e.g. no visible GPU on torch 2.6).
    try:
        import torch  # noqa: F401

        from .models.hf_causal import _shim_torch_accelerator

        _shim_torch_accelerator(torch)
        from transformers import AutoModelForCausalLM  # noqa: F401

        print("  transformers model import: ok")
    except Exception as exc:  # noqa: BLE001
        print(f"  transformers model import: FAILED ({type(exc).__name__}: {exc})")

    try:
        import torch

        avail = torch.cuda.is_available()
        print(f"  cuda available: {avail}")
        if avail:
            print(f"  cuda version: {torch.version.cuda}")
            print(f"  gpu count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
        else:
            print("  note: no GPU visible to torch; set a valid CUDA_VISIBLE_DEVICES "
                  "if you expected one (a torch/transformers version mismatch can "
                  "also break the transformers import when no accelerator is present)")
    except Exception:  # noqa: BLE001
        print("  cuda available: unknown (torch not installed)")

    import os

    for var in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES", "HF_TOKEN"):
        val = os.environ.get(var)
        if val and var == "HF_TOKEN":
            val = "set (hidden)"
        print(f"  {var}={val}")

    from .models.base import available_models

    print(f"  model adapters: {', '.join(available_models())}")

    try:
        import torch  # noqa: F401

        ready = True
    except Exception:  # noqa: BLE001
        ready = False
    print("  status: " + ("ready for real models" if ready else
                          "core OK; install the [hf] extra to run real models"))
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
        adapter=args.adapter,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_context_tokens=args.max_context_tokens,
        chat_template=_read_maybe_path(args.chat_template),
        attn_implementation=args.attn,
        adapter_extra=({"max_batch_tokens": args.max_batch_tokens}
                       if getattr(args, "max_batch_tokens", None) else {}),
    )
    summary = run_smoke(cfg)
    print(render_smoke(summary))
    return 0 if summary["ok"] else 1


def _build_run_config(args):
    from .eval.runner import RunConfig
    from .prompts.options import PromptOptions

    noul_words = ("true", "false")
    if getattr(args, "noul_words", None):
        parts = [p.strip() for p in args.noul_words.split(",")]
        if len(parts) == 2:
            noul_words = (parts[0], parts[1])
    prompt = PromptOptions(
        system_prompt=args.system_prompt,
        answer_prefix=args.answer_prefix or "",
        state_fit=args.state_fit,
        noul_words=noul_words,
        label_options=not getattr(args, "no_label_options", False),
        continuation_lead_space=not getattr(args, "no_lead_space", False),
    )
    adapter_extra = {}
    if getattr(args, "max_batch_tokens", None):
        adapter_extra["max_batch_tokens"] = args.max_batch_tokens
    if getattr(args, "enable_thinking", False):
        adapter_extra["chat_template_kwargs"] = {"enable_thinking": True}
    return RunConfig(
        data_path=args.data,
        adapter=args.adapter,
        model_id=args.model_id,
        sources=_csv(args.sources),
        question_types=_csv(args.types),
        splits=_csv(args.split),
        limit=args.limit,
        per_source_limit=args.per_source_limit,
        sample=args.sample,
        seed=args.seed,
        score_mode=(None if args.score_mode == "auto" else args.score_mode),
        normalize=args.normalize,
        temperature=args.temperature,
        batch_size=args.batch_size,
        top_k=args.top_k,
        device=args.device,
        dtype=args.dtype,
        max_context_tokens=args.max_context_tokens,
        chat_template=_read_maybe_path(args.chat_template),
        attn_implementation=args.attn,
        output_dir=args.output_dir,
        run_name=args.run_name,
        prompt=prompt,
        adapter_extra=adapter_extra,
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


def cmd_report(args) -> int:
    from .eval.report import report_from_predictions, render_markdown

    config = None
    if args.adapter or args.model_id or args.score_mode:
        config = {
            "adapter": args.adapter,
            "model_id": args.model_id,
            "score_mode": args.score_mode,
            "normalize": None,
            "temperature": None,
        }
    summary = report_from_predictions(
        args.predictions, output_dir=args.output_dir,
        run_name=args.run_name, config=config,
    )
    print(render_markdown(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="menoj", description="Loglikelihood JEV evaluator")
    sub = p.add_subparsers(dest="command", required=True)

    pm = sub.add_parser("models", help="list available model adapters")
    pm.set_defaults(func=cmd_models)

    pd = sub.add_parser("doctor", help="show the environment (python, torch, CUDA, env vars)")
    pd.set_defaults(func=cmd_doctor)

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
    pk.add_argument("--modes", default="auto,continuation,next_token")
    pk.add_argument("--sample", choices=["head", "random"], default="head")
    pk.add_argument("--seed", type=int, default=0)
    pk.add_argument("--adapter", default="mock", help="mock (default), hf, qwen, ...")
    pk.add_argument("--model-id", default=None, help="local path or HF hub id")
    pk.add_argument("--device", default=None)
    pk.add_argument("--dtype", default="auto")
    pk.add_argument("--max-context-tokens", type=int, default=None)
    pk.add_argument("--chat-template", default=None, help="path to a template file or literal")
    pk.add_argument("--attn", default=None)
    pk.add_argument("--max-batch-tokens", type=int, default=None, help="tokens per forward pass (default: auto from GPU memory)")
    pk.set_defaults(func=cmd_smoke)

    prep = sub.add_parser("report", help="(re)compute metrics from a saved predictions.jsonl")
    prep.add_argument("predictions", help="path to a *.predictions.jsonl file")
    prep.add_argument("--output-dir", default=None, help="where to write the report (default: alongside predictions)")
    prep.add_argument("--run-name", default=None)
    prep.add_argument("--adapter", default=None)
    prep.add_argument("--model-id", default=None)
    prep.add_argument("--score-mode", default=None)
    prep.set_defaults(func=cmd_report)

    pr = sub.add_parser("run", help="run the evaluation")
    pr.add_argument("--data", default="benchmark.json")
    pr.add_argument("--adapter", default="hf", help="model adapter name (hf, qwen, mock, ...)")
    pr.add_argument("--model-id", default=None, help="local path or HF hub id")
    pr.add_argument("--sources", default=None, help="comma-separated source filter")
    pr.add_argument("--types", default=None, help="comma-separated question types")
    pr.add_argument("--split", default=None, help="comma-separated splits (train,dev,test)")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--per-source-limit", type=int, default=None)
    pr.add_argument("--sample", type=int, default=None,
                    help="draw a small stratified subsample of N items (quick eval)")
    pr.add_argument("--seed", type=int, default=0, help="sampling seed")
    pr.add_argument("--score-mode", choices=["auto", "continuation", "next_token"], default=None,
                    help="scoring mode; default auto picks per item by answer shape")
    pr.add_argument("--normalize", choices=["length", "sum", "none"], default="length")
    pr.add_argument("--temperature", type=float, default=1.0)
    pr.add_argument("--batch-size", type=int, default=None,
                    help="candidates scored per call (default: auto by token budget)")
    pr.add_argument("--top-k", type=int, default=5)
    pr.add_argument("--device", default=None)
    pr.add_argument("--dtype", default="auto")
    pr.add_argument("--max-context-tokens", type=int, default=None)
    pr.add_argument("--chat-template", default=None, help="path to a template file or literal")
    pr.add_argument("--attn", default=None, help="attn_implementation (e.g. sdpa, flash_attention_2)")
    pr.add_argument("--system-prompt", default=None)
    pr.add_argument("--answer-prefix", default=None)
    pr.add_argument("--noul-words", default=None, help="binary answer words, e.g. 'true,false' or 'yes,no'")
    pr.add_argument("--no-label-options", action="store_true", help="do not letter-label options A/B/C")
    pr.add_argument("--no-lead-space", action="store_true", help="do not prefix continuations with a space")
    pr.add_argument("--enable-thinking", action="store_true",
                    help="Qwen: keep thinking mode ON (default: OFF)")
    pr.add_argument("--state-fit", choices=["head", "tail", "head_tail", "none"], default="head_tail")
    pr.add_argument("--max-batch-tokens", type=int, default=None, help="tokens per forward pass (default: auto from GPU memory)")
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
