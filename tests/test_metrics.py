from types import SimpleNamespace

from jev.eval.metrics import summarize


def _p(i, source, correct, prob, rank, error=None, qtype="choice", split="test"):
    return SimpleNamespace(
        id=f"{source}:{i}", source=source, question_type=qtype, split=split,
        gold_id="g", pred_id="g" if correct else "x", correct=correct,
        prob_gold=prob, rank_gold=rank, score_mode="next_token", error=error,
        top=[],
    )


def test_errors_excluded_from_accuracy():
    preds = [
        _p(0, "a", True, 0.9, 1),
        _p(1, "a", False, 0.3, 2),
        _p(2, "a", False, 0.0, None, error="OutOfMemoryError: OOM"),
        _p(3, "a", False, 0.0, None, error="OutOfMemoryError: OOM"),
    ]
    s = summarize(preds)
    # 4 items, 2 scored, 1 of them correct -> accuracy 0.5 (NOT 0.25)
    assert s["overall"]["n"] == 4
    assert s["overall"]["scored"] == 2
    assert s["overall"]["errors"] == 2
    assert abs(s["overall"]["accuracy"] - 0.5) < 1e-9
    assert s["error_count"] == 2
    assert s["error_types"].get("OutOfMemoryError") == 2


def test_fully_errored_source_does_not_enter_macro():
    preds = [
        _p(0, "a", True, 0.9, 1),
        _p(1, "b", False, 0.0, None, error="OutOfMemoryError: OOM"),
    ]
    s = summarize(preds)
    src_a = s["by_source"]["a"]
    src_b = s["by_source"]["b"]
    assert src_a["scored"] == 1 and abs(src_a["accuracy"] - 1.0) < 1e-9
    assert src_b["scored"] == 0 and src_b["errors"] == 1
    # macro over types ignores fully-errored buckets (here only 'choice', 1.0)
    assert abs(s["macro_accuracy_by_type"] - 1.0) < 1e-9
