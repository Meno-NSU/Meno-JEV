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
        return f"| {name} | 0 | 0 | - | - | - | 0 |"
    err = m.get("errors", 0)
    scored = m.get("scored", m.get("n", 0) - err)
    acc = f"{m['accuracy']:.4f}" if scored else "-"
    mp = f"{m.get('mean_prob_gold', 0.0):.4f}" if scored else "-"
    mrr = f"{m.get('mrr', 0.0):.4f}" if scored else "-"
    return f"| {name} | {m['n']} | {scored} | {acc} | {mp} | {mrr} | {err} |"


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    name = summary.get("run_name", "run")
    lines.append(f"# Meno-JEV run: {name}")
    cfg = summary.get("config") or {}
    if cfg:
        lines.append("")
        lines.append(
            f"- adapter: `{cfg.get('adapter') or '-'}`  "
            f"model: `{cfg.get('model_id') or '-'}`"
        )
        mode = cfg.get("score_mode") or "auto"
        lines.append(
            f"- score_mode: `{mode}`  normalize: `{cfg.get('normalize')}`"
            f"  temperature: `{cfg.get('temperature')}`"
        )
    ov = summary["overall"]
    lines.append("")
    lines.append(
        f"**Overall accuracy: {ov.get('accuracy', 0.0):.4f}** "
        f"(scored={ov.get('scored', ov.get('n', 0))}, n={ov.get('n', 0)}, "
        f"errors={ov.get('errors', 0)})"
    )
    lines.append(
        "Accuracy / mean P(gold) / MRR are computed over **scored** items only; "
        "failed items (errors) are excluded, not counted as wrong."
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
    lines.append("| type | n | scored | accuracy | mean P(gold) | MRR | errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in sorted(summary.get("by_question_type", {})):
        lines.append(_row(k, summary["by_question_type"][k]))

    lines.append("\n## By source\n")
    lines.append("| source | n | scored | accuracy | mean P(gold) | MRR | errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
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
        f"**{n_err} item(s) failed** (mostly CUDA OOM). These are excluded from "
        "accuracy / mean P(gold) / MRR — they are infrastructure failures, not "
        "wrong model answers. The `scored` column shows how many items each group "
        "actually used."
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


def report_from_predictions(
    predictions_path,
    output_dir=None,
    run_name=None,
    config=None,
) -> Dict[str, Any]:
    """Recompute metrics from a saved ``*.predictions.jsonl`` (no model needed).

    Reuses the ``config``/``run_name`` from a sibling ``<run>.report.json`` when
    present. Writes ``<run>.report.{json,md}`` and returns the summary.
    """
    from .metrics import summarize
    from .preds import load_predictions

    p = Path(predictions_path)
    name = run_name or p.name.replace(".predictions.jsonl", "")

    if config is None:
        sibling = p.with_name(f"{name}.report.json")
        if sibling.exists():
            try:
                config = json.loads(sibling.read_text(encoding="utf-8")).get("config")
            except (ValueError, OSError):
                config = None

    summary = summarize(load_predictions(p), cfg=config)
    summary["run_name"] = name
    summary["predictions_path"] = str(p)

    out = Path(output_dir) if output_dir else p.parent
    write_json(summary, out / f"{name}.report.json")
    write_markdown(summary, out / f"{name}.report.md")
    return summary
