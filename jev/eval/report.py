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
        return f"| {name} | 0 | - | - | - | - |"
    err = m.get("errors", 0)
    return (
        f"| {name} | {m['n']} | {m['accuracy']:.4f} | "
        f"{m.get('mean_prob_gold', 0.0):.4f} | {m.get('mrr', 0.0):.4f} | {err} |"
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
        mode = cfg.get("score_mode") or "auto"
        lines.append(
            f"- score_mode: `{mode}`  normalize: `{cfg.get('normalize')}`"
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
    if summary.get("oom_retries"):
        lines.append(
            f"\n> CUDA OOM retries during the run: {summary['oom_retries']} "
            "(batches were shrunk automatically; lower `--max-batch-tokens` / "
            "`--max-context-tokens` to avoid it)."
        )

    lines.append("\n## By question type\n")
    lines.append("| type | n | accuracy | mean P(gold) | MRR | errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k in sorted(summary.get("by_question_type", {})):
        lines.append(_row(k, summary["by_question_type"][k]))

    lines.append("\n## By source\n")
    lines.append("| source | n | accuracy | mean P(gold) | MRR | errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k in sorted(summary.get("by_source", {})):
        lines.append(_row(k, summary["by_source"][k]))

    lines.extend(_errors_section(summary))
    return "\n".join(lines) + "\n"


def _errors_section(summary: Dict[str, Any]) -> list:
    n_err = summary.get("error_count", 0)
    lines = ["\n## Errors\n"]
    if not n_err:
        lines.append("None.")
        return lines
    lines.append(
        f"**{n_err} item(s) failed** — these count as wrong (accuracy 0, "
        "mean P(gold) 0, MRR 0), so they drag the averages down."
    )
    types = summary.get("error_types") or {}
    if types:
        lines.append("")
        lines.append("By type: " + ", ".join(f"`{k}` x{v}" for k, v in types.items()))
    details = summary.get("error_details") or []
    if details:
        lines.append("")
        lines.append("| id | source | type | mode | error |")
        lines.append("| --- | --- | --- | --- | --- |")
        for d in details:
            msg = (d.get("error") or "").replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {d['id']} | {d['source']} | {d['question_type']} | "
                f"{d.get('mode', '')} | {msg[:200]} |"
            )
    return lines


def write_markdown(summary: Dict[str, Any], path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(summary), encoding="utf-8")
    return p
