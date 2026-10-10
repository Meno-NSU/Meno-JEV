import os

import pytest

from jev.eval.smoke import SmokeConfig, check_item, run_smoke
from jev.data.schema import parse_item
from jev.models.base import ModelConfig, build_model
from jev.prompts.options import PromptOptions

REAL_BENCHMARK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark.json")


def test_smoke_on_tiny_fixture(tiny_path):
    summary = run_smoke(SmokeConfig(data_path=tiny_path, per_source_limit=10))
    assert summary["ok"], summary["failures"]
    assert summary["checks_run"] == 6  # 3 items x 2 modes
    assert summary["checks_failed"] == 0


def test_smoke_detects_missing_gold(tiny_records):
    adapter = build_model("mock", ModelConfig(), load=True)
    item = parse_item(tiny_records[0])
    item.gold_id = "does-not-exist"
    check = check_item(item, adapter, "continuation", PromptOptions())
    assert not check.ok
    assert any("gold id" in m for m in check.issues)


@pytest.mark.skipif(not os.path.exists(REAL_BENCHMARK), reason="benchmark.json not present")
def test_smoke_on_real_benchmark():
    summary = run_smoke(SmokeConfig(data_path=REAL_BENCHMARK, per_source_limit=2))
    assert summary["checks_failed"] == 0, summary["failures"][:10]
    assert summary["sources"] >= 50
    assert set(summary["question_types_seen"]) == {"choice", "noul", "score"}
