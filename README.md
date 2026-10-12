# Meno-JEV

Loglikelihood ("JEV"-style) evaluation: turn any causal LLM into a **scorer**.

Instead of letting a model *generate* an answer, Meno-JEV conditions the model on
the **state** (context) and the **instruction** (question), then measures the
probability of each candidate answer from the **criteria** (answer space). The
model never writes free text; it only re-weights the given options. The result is
a probability distribution over candidates, and the prediction is its argmax.

```
item (state, instruction, criteria, gold)
        │
        ▼  recipe.build(...)
prompt up to the answer position  ──►  model.loglikelihood / next-token logits
        │                                             │
        ▼  scoring.modes.score(...)                    ▼
softmax over candidates  ──►  distribution  ──►  prediction + metrics
```

## Install

Full step-by-step (venv/conda, CUDA-matched torch, HF cache/offline, env vars,
troubleshooting) is in **[SETUP.md](SETUP.md)**. The short version:

```bash
bash scripts/setup_env.sh                 # core (no torch; runs the mock)
WITH_HF=1 bash scripts/setup_env.sh       # + torch/transformers for real models
source .venv/bin/activate

python -m jev.cli doctor                  # verify the environment
```

Manual: `pip install -e .` (core) and `pip install -e ".[hf]"` after installing
the torch build that matches your CUDA (see SETUP.md §3).

## Quick start

```bash
# see registered model adapters
python -m jev.cli models

# inspect the dataset (sources, counts, types)
python -m jev.cli sources --data benchmark.json

# eyeball how an item is parsed
python -m jev.cli inspect --data benchmark.json --source winogrande --limit 2

# self-check the harness on a small subsample of every source (exit != 0 on failure)
python -m jev.cli smoke --data benchmark.json

# run a real model in loglikelihood mode
python -m jev.cli run --data benchmark.json \
    --adapter qwen --model-id /path/to/qwen-finetune \
    --sources mmlu-pro,med-qa,winogrande --types choice \
    --score-mode continuation --normalize length \
    --output-dir outputs --run-name qwen_tune_v1
```

Outputs (per run) land in `outputs/`:

- `<run>.predictions.jsonl` - one JSON object per item (gold, prediction, prob, rank, top-k).
- `<run>.report.json` - full summary (overall / by type / by source / by split).
- `<run>.report.md` - readable accuracy table.

Items that fail (e.g. CUDA OOM) are **excluded** from accuracy / mean P(gold) /
MRR — an infrastructure failure is not a wrong answer. The report shows `n`,
`scored` and `errors` per group (and an `## Errors` section), so coverage stays
visible. Accuracy is always over the **scored** items.

`benchmark.json` is intentionally **not** committed (113 MB); pass `--data`.

## Smoke check

`menoj smoke` verifies the **harness**, not model quality. Using the
dependency-free `mock` adapter it samples `--per-source-limit` items from every
source, runs both scoring modes, and checks the structural invariants:

- every source parses and resolves a recipe;
- each rendered query has candidates and a gold candidate present among them;
- candidate counts match the raw criteria (choice dict / noul=2 / score list);
- the distribution is a valid probability vector over exactly the candidates;
- no item raises an error.

```bash
python -m jev.cli smoke --data benchmark.json --per-source-limit 4
# -> checks: 464  failed: 0 ... RESULT: OK   (exit 0; exit 1 if anything is off)
```

By default `smoke` uses the `mock` adapter (no GPU/model needed) so it validates
the harness itself. To smoke-test a **real** model (that it loads and scores on a
few items), pass the same model flags as `run`:

```bash
python -m jev.cli smoke --data benchmark.json --per-source-limit 1 \
    --adapter qwen --model-id /path/to/your-finetune
```

It is also a pytest: `tests/test_smoke.py` runs it against a tiny fixture always
and against the real `benchmark.json` when the file is present (skipped otherwise).

## Quick eval (prompt tuning)

Iterate on the prompt without running all ~11k items. `--sample N` draws a small,
**reproducible, stratified** subsample: per-question-type counts follow the real
mix (≈85% choice / 12% noul / 3% score) and items are spread round-robin across
sources, so ~100 items still touch all sources.

```bash
python -m jev.cli run --data benchmark.json --adapter qwen \
    --model-id /path/to/your-finetune \
    --sample 100 --seed 0 \
    --output-dir outputs --run-name tune-sample100
# n=100, ~85 choice / 12 noul / 3 score, all 58 sources
```

Rotate the prompt with these flags (all also work on the full run):

| flag | effect |
| --- | --- |
| `--system-prompt TEXT` | prepend a system message |
| `--answer-prefix TEXT` | text inserted right before the scored continuation |
| `--noul-words "true,false"` | binary words (`yes,no`, `1,0`, ...) |
| `--no-label-options` | don't list options as `A. ...` |
| `--no-lead-space` | don't prefix continuations with a space |
| `--score-mode` / `--normalize` / `--temperature` | scoring knobs |
| `--max-batch-tokens` / `--batch-size` | forward batching (default auto from GPU memory) |
| `--state-fit` / `--max-context-tokens` | how long states are truncated |

> **Qwen thinking is OFF by default.** Qwen3/Qwen3.5 would otherwise open a
> reasoning block right after the assistant turn, so the first scored token would
> be reasoning instead of the answer. The `qwen` adapter renders the chat
> template with `enable_thinking=False`. Pass `--enable-thinking` to keep it on.

## Scoring methods

Two strategies. `--score-mode` defaults to **`auto`**, which picks per item from
the shape of the answer space:

- **choice with ≤26 options** → `next_token` (scores the option letter A/B/C…,
  matching how options are listed);
- **noul / score whose labels are single distinct tokens** (`true`/`false`,
  `1..5`) → `next_token`;
- **long/named options or >26 candidates** (entity, intent, big sets) →
  `continuation`;
- override with `--score-mode continuation` or `--score-mode next_token`.

The two modes themselves:

| mode | how | good for |
| --- | --- | --- |
| `continuation` (default) | Append the candidate text after the answer prefix and sum its token log-probabilities; `--normalize length` divides by token count and softens length bias | free-text options of any length (medical, legal, entity names) |
| `next_token` | Read the log-probabilities of the first token of each candidate (a letter, a digit, `true`/`false`) | compact answer spaces labeled A/B/C or numeric scales |

`--normalize {length,sum,none}` and `--temperature` control the softmax. Long
states are truncated into the model context with `--state-fit {head,tail,head_tail,none}`.

## Architecture

```
jev/
  data/      schema.py (parse state/instruction/criteria/gold), loader.py (load+filter)
  prompts/   options.py (knobs), format.py (render state, options, context, truncation)
  models/    base.py (ModelAdapter ABC + registry), registry.py (auto-import)
             hf_causal.py (generic HF), qwen.py (Qwen), mock.py (no-dependency)
  scoring/   distribution.py (softmax, Distribution), modes.py (continuation/next_token)
  tasks/     base.py (Recipe + registry), generic.py (choice/noul/score)
             sources/ (per-source overrides, auto-imported; _template.py example)
  eval/      runner.py, metrics.py, report.py
  cli.py
```

The three question types map to a uniform candidate list:

- **choice** - `criteria` is a dict `{key: text}`; gold is a key.
- **noul** - binary; gold `1`/`0` -> `true`/`false`.
- **score** - `criteria` is a list; gold is a 0-based index.

### Adding a new model family (isolated by design)

Nothing existing changes: add one file, it self-registers.

```python
# jev/models/myfamily.py
from .hf_causal import HFCausalAdapter
from .base import register_model

@register_model("myfamily")
class MyFamilyAdapter(HFCausalAdapter):
    def load(self):
        super().load()
        # family-specific tweaks: chat template, pad token, prompt prefix, ...
```

Run it with `--adapter myfamily --model-id ...`. Files named `_*.py` are ignored
(use them as templates).

### Special-casing a source

Drop a file into `jev/tasks/sources/` (copy `sources/_template.py`). Register a
`Recipe` with `priority > -100`; it takes over only for the items its `matches`
accepts. The generic recipe keeps handling everything else.

## Testing

```bash
pytest            # runs fully on CPU using the `mock` adapter, no torch/models needed
```

The mock adapter accepts `ModelConfig.extra = {"preferred_substrings": [...]}`
or a `scores_fn(context, continuation) -> float` for deterministic tests.
