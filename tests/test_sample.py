from jev.data.loader import sample_items
from jev.data.schema import Item


def _mk(i, source, qtype):
    return Item(id=f"{source}:{i}", source=source, state="s", question_type=qtype,
                source_item_id=f"{source}/test/{i}")


def test_sample_size_and_determinism():
    items = [_mk(i, f"src{i % 5}", "choice") for i in range(100)]
    a = sample_items(items, 20, seed=1)
    b = sample_items(items, 20, seed=1)
    assert len(a) == 20
    assert [x.id for x in a] == [x.id for x in b]  # reproducible


def test_sample_is_stratified_and_spreads_sources():
    items = []
    for i in range(200):
        items.append(_mk(i, "big", "choice"))
    for i in range(20):
        items.append(_mk(i, "noul-src", "noul"))
    for i in range(5):
        items.append(_mk(i, "score-src", "score"))
    out = sample_items(items, 50, seed=0)
    types = {x.question_type for x in out}
    assert types == {"choice", "noul", "score"}  # every type represented
    assert len({x.source for x in out}) >= 3      # sources spread


def test_sample_noop_when_large():
    items = [_mk(i, "s", "choice") for i in range(10)]
    assert len(sample_items(items, 50)) == 10
