"""Write a run summary to JSON and a human-readable Markdown table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(summary: Dict[str, Any], path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _row(name: str, m: Dict[str, Any]) -> str:
    if not m or m.get("n", 0) == 0:
        return f"| {name} | 0 | - | - | - |"
    return (
        f"| {name} | {m['n']} | {m['accuracy']:.4f} | "
        f"{m.get('mean_prob_gold', 0.0):.4f} | {m.get('mrr', 0.0):.4f} |"
    )


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    name = summary.get("run_name", "run")
    lines.append(f"# Meno-JEV run: {name}")
    cfg = summary.get("config") or {}
    if cfg:
        lines.append("")
        lines.append(
            f"- adapter: `{cfg.get('adapter')}`  model: `{cfg.get('model_id')}`"
        )
        lines.append(
            f"- score_mode: `{cfg.get('score_mode')}`  normalize: `{cfg.get('normalize')}`"
            f"  temperature: `{cfg.get('temperature')}`"
        )
    lines.append("")
    lines.append(
        f"**Overall accuracy: {summary['overall'].get('accuracy', 0.0):.4f} "
        f"(n={summary['overall'].get('n', 0)})**"
    )
    lines.append(
        f"Macro accuracy by question type: {summary.get('macro_accuracy_by_type', 0.0):.4f}"
    )

    lines.append("\n## By question type\n")
    lines.append("| type | n | accuracy | mean P(gold) | MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for k in sorted(summary.get("by_question_type", {})):
        lines.append(_row(k, summary["by_question_type"][k]))

    lines.append("\n## By source\n")
    lines.append("| source | n | accuracy | mean P(gold) | MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for k in sorted(summary.get("by_source", {})):
        lines.append(_row(k, summary["by_source"][k]))

    return "\n".join(lines) + "\n"


def write_markdown(summary: Dict[str, Any], path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(summary), encoding="utf-8")
    return p
