from jev.cli import build_parser


def _cfg(argv):
    from jev.cli import _build_run_config

    ns = build_parser().parse_args(argv)
    return _build_run_config(ns)


def test_safe_defaults():
    cfg = _cfg(["run", "--safe", "--adapter", "qwen", "--model-id", "m"])
    assert cfg.max_context_tokens == 8192
    assert cfg.batch_size == 1
    assert cfg.dtype == "bfloat16"
    assert cfg.adapter_extra.get("max_batch_tokens") == 2048


def test_safe_does_not_override_explicit():
    cfg = _cfg([
        "run", "--safe", "--adapter", "qwen", "--model-id", "m",
        "--max-context-tokens", "4096", "--batch-size", "4",
        "--dtype", "float16", "--max-batch-tokens", "1024",
    ])
    assert cfg.max_context_tokens == 4096
    assert cfg.batch_size == 4
    assert cfg.dtype == "float16"
    assert cfg.adapter_extra.get("max_batch_tokens") == 1024


def test_no_safe_keeps_defaults():
    cfg = _cfg(["run", "--adapter", "qwen", "--model-id", "m"])
    assert cfg.max_context_tokens is None
    assert cfg.batch_size is None
    assert cfg.dtype == "auto"
