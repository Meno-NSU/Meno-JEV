import json

from jev.cli import main
from jev.eval.preds import failed_ids, merge_predictions


def _row(item_id, source, correct, error=None):
    return {
        "id": item_id, "source": source, "question_type": "noul", "split": "test",
        "gold_id": "true", "pred_id": "true" if correct else "",
        "correct": correct, "prob_gold": 0.9 if correct else 0.0,
        "rank_gold": 1 if correct else None, "score_mode": "next_token",
        "top": [], "error": error,
    }


def test_failed_ids_and_merge(tmp_path):
    base = tmp_path / "base.predictions.jsonl"
    retry = tmp_path / "base_retry.predictions.jsonl"
    with open(base, "w") as fh:
        fh.write(json.dumps(_row("a:0", "a", True)) + "\n")
        fh.write(json.dumps(_row("a:1", "a", False, "OutOfMemoryError: x")) + "\n")
    with open(retry, "w") as fh:
        fh.write(json.dumps(_row("a:1", "a", True)) + "\n")  # now fixed

    assert failed_ids(base) == ["a:1"]
    merged = merge_predictions(base, retry, tmp_path / "base_merged.predictions.jsonl")
    by_id = {p.id: p for p in merged}
    assert len(merged) == 2
    assert by_id["a:1"].error is None
    assert by_id["a:1"].correct is True


def test_retry_command_fixes_errors(tiny_path, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    base = out / "base.predictions.jsonl"
    # one good item, one failed item (both exist in the tiny benchmark)
    with open(base, "w") as fh:
        fh.write(json.dumps({
            "id": "unit:choice:0", "source": "unit", "question_type": "choice",
            "split": "test", "gold_id": "opt_2", "pred_id": "opt_1",
            "correct": False, "prob_gold": 0.2, "rank_gold": 2,
            "score_mode": "next_token", "top": [], "error": None,
        }) + "\n")
        fh.write(json.dumps({
            "id": "unit:noul:1", "source": "unit", "question_type": "noul",
            "split": "test", "gold_id": "true", "pred_id": "",
            "correct": False, "prob_gold": 0.0, "rank_gold": None,
            "score_mode": "next_token", "top": [],
            "error": "OutOfMemoryError: CUDA out of memory",
        }) + "\n")

    rc = main([
        "retry", str(base), "--data", tiny_path, "--adapter", "mock",
        "--output-dir", str(out), "--run-name", "base_retry",
    ])
    assert rc == 0
    merged_path = out / "base_merged.predictions.jsonl"
    assert merged_path.exists()
    rows = [json.loads(l) for l in merged_path.read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}
    assert len(rows) == 2
    assert by_id["unit:noul:1"]["error"] is None  # retried successfully
