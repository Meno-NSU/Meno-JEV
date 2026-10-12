import json

from jev.eval.preds import load_predictions
from jev.eval.report import report_from_predictions


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_load_and_recompute_from_predictions(tmp_path):
    rows = [
        {"id": "a:0", "source": "a", "question_type": "choice", "split": "test",
         "gold_id": "g", "pred_id": "g", "correct": True, "prob_gold": 0.9,
         "rank_gold": 1, "score_mode": "next_token", "top": [], "error": None},
        {"id": "a:1", "source": "a", "question_type": "choice", "split": "test",
         "gold_id": "g", "pred_id": "x", "correct": False, "prob_gold": 0.3,
         "rank_gold": 2, "score_mode": "next_token", "top": [], "error": None},
        {"id": "b:0", "source": "b", "question_type": "choice", "split": "test",
         "gold_id": "g", "pred_id": "", "correct": False, "prob_gold": 0.0,
         "rank_gold": None, "score_mode": "next_token", "top": [],
         "error": "OutOfMemoryError: OOM"},
    ]
    path = tmp_path / "run.predictions.jsonl"
    _write(path, rows)

    preds = load_predictions(path)
    assert len(preds) == 3

    summary = report_from_predictions(str(path))
    # 3 items, 1 error -> 2 scored, 1 correct -> accuracy 0.5 (not 1/3)
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["scored"] == 2
    assert summary["overall"]["errors"] == 1
    assert abs(summary["overall"]["accuracy"] - 0.5) < 1e-9
    # report files written next to predictions
    assert (tmp_path / "run.report.json").exists()
    assert (tmp_path / "run.report.md").exists()
