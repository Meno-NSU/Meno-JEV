from jev.eval.runner import RunConfig, run
from jev.models.base import ModelConfig, build_model
from jev.prompts.options import PromptOptions


def _mock(preferred):
    return build_model(
        "mock", ModelConfig(extra={"preferred_substrings": list(preferred)}), load=True
    )


def test_continuation_pipeline_picks_preferred(tiny_path, tmp_path):
    cfg = RunConfig(
        data_path=tiny_path,
        adapter="mock",
        output_dir=str(tmp_path / "out"),
        run_name="cont",
        prompt=PromptOptions(),
    )
    # "beta" is the gold text of unit:choice:0; "true" and "5" for the others.
    adapter = _mock(["beta", "true", "5"])
    summary = run(cfg, adapter=adapter)
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["accuracy"] == 1.0


def test_continuation_pipeline_when_wrong(tiny_path, tmp_path):
    cfg = RunConfig(
        data_path=tiny_path,
        adapter="mock",
        output_dir=str(tmp_path / "out"),
        run_name="wrong",
        prompt=PromptOptions(),
    )
    adapter = _mock(["alpha"])  # prefers the wrong choice option
    summary = run(cfg, adapter=adapter)
    # choice wrong (alpha); score ties -> first option wrong; noul ties -> "true" correct
    assert summary["by_source"]["unit"]["n"] == 3
    assert abs(summary["overall"]["accuracy"] - 1 / 3) < 1e-9


def test_next_token_mode(tiny_path, tmp_path):
    cfg = RunConfig(
        data_path=tiny_path,
        adapter="mock",
        score_mode="next_token",
        output_dir=str(tmp_path / "out"),
        run_name="nt",
        prompt=PromptOptions(),
    )
    # choice gold is opt_2 -> letter "B"; noul gold "true"; score gold index 4 -> "5"
    adapter = _mock(["B", "true", "5"])
    summary = run(cfg, adapter=adapter)
    assert summary["overall"]["accuracy"] == 1.0


def test_filters(tiny_path, tmp_path):
    cfg = RunConfig(
        data_path=tiny_path,
        adapter="mock",
        question_types=["choice"],
        output_dir=str(tmp_path / "out"),
        run_name="filt",
        prompt=PromptOptions(),
    )
    adapter = _mock(["beta"])
    summary = run(cfg, adapter=adapter)
    assert summary["overall"]["n"] == 1
