import math

from jev.scoring.distribution import Distribution, softmax


def test_softmax_normalized():
    p = softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[2] > p[1] > p[0]


def test_softmax_stable():
    p = softmax([1000.0, 1000.0])
    assert abs(sum(p) - 1.0) < 1e-9
    assert abs(p[0] - 0.5) < 1e-9


def test_distribution_rank_and_prob():
    d = Distribution(
        ids=["a", "b", "c"],
        scores=[0.1, 0.9, 0.5],
        probs=softmax([0.1, 0.9, 0.5]),
    )
    assert d.argmax_id() == "b"
    assert d.rank_of("b") == 1
    assert d.rank_of("c") == 2
    assert d.rank_of("a") == 3
    assert abs(sum(d.as_dict().values()) - 1.0) < 1e-9
    assert d.prob_of("missing") == 0.0


def test_softmax_temperature():
    hot = softmax([1.0, 2.0], temperature=0.5)
    cold = softmax([1.0, 2.0], temperature=2.0)
    assert hot[1] > cold[1]
