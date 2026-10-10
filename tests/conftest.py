import json

import pytest


def _record(id_, source, state, qtype, questions, label, item_id):
    return {
        "id": id_,
        "source": source,
        "state": state,
        "question_type": qtype,
        "questions_json": json.dumps(questions),
        "label_json": json.dumps(label),
        "source_item_id": item_id,
    }


@pytest.fixture
def tiny_records():
    return [
        _record(
            "unit:choice:0", "unit", "Some context about a question.",
            "choice",
            {"q": {"type": "choice", "instructions": "Pick one.",
                   "criteria": {"opt_1": "alpha", "opt_2": "beta", "opt_3": "gamma"}}},
            "opt_2", "unit/test/0",
        ),
        _record(
            "unit:noul:1", "unit", "Evidence text.",
            "noul",
            {"q": {"type": "noul", "instructions": "Is it supported?",
                   "criteria": {"true": "supported", "false": "not supported"}}},
            1, "unit/test/1",
        ),
        _record(
            "unit:score:2", "unit", "Rate quality.",
            "score",
            {"q": {"type": "score", "instructions": "Rate 1-5.",
                   "criteria": [1, 2, 3, 4, 5]}},
            4, "unit/test/2",
        ),
    ]


@pytest.fixture
def tiny_path(tmp_path, tiny_records):
    p = tmp_path / "benchmark.json"
    p.write_text(json.dumps(tiny_records), encoding="utf-8")
    return str(p)
