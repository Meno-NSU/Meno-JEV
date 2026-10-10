from jev.data.schema import parse_item


def test_parse_choice(tiny_records):
    item = parse_item(tiny_records[0])
    assert item.question_type == "choice"
    assert [c.id for c in item.candidates] == ["opt_1", "opt_2", "opt_3"]
    assert item.gold_id == "opt_2"
    assert item.gold_candidate().text == "beta"
    # A per-option letter is available as the short token.
    assert [c.token for c in item.candidates] == ["A", "B", "C"]


def test_parse_noul(tiny_records):
    item = parse_item(tiny_records[1])
    assert [c.id for c in item.candidates] == ["true", "false"]
    assert item.gold_id == "true"
    assert item.split == "test"


def test_parse_score(tiny_records):
    item = parse_item(tiny_records[2])
    assert [c.id for c in item.candidates] == ["0", "1", "2", "3", "4"]
    assert item.gold_id == "4"
    assert item.gold_candidate().text == "5"


def test_noul_false_label():
    rec = {
        "id": "x", "source": "unit", "state": "s", "question_type": "noul",
        "questions_json": '{"q": {"type": "noul", "criteria": {}}}',
        "label_json": "0", "source_item_id": "unit/test/0",
    }
    assert parse_item(rec).gold_id == "false"


def test_named_choice_keys():
    rec = {
        "id": "x", "source": "entity", "state": "s", "question_type": "choice",
        "questions_json": '{"entity": {"type": "choice", "criteria": {"Paris": null, "Rome": null}}}',
        "label_json": '"Paris"', "source_item_id": "entity/test/0",
    }
    item = parse_item(rec)
    assert [c.id for c in item.candidates] == ["Paris", "Rome"]
    assert item.candidates[0].text == "Paris"  # null value falls back to key
    assert item.gold_id == "Paris"
