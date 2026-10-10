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

```bash
pip install -e .                 # core (no torch needed; works with the mock)
pip install -e ".[hf]"           # + torch/transformers for real models
```

On the cluster (real model):

```bash
pip install torch transformers accelerate sentencepiece
```

## Quick start

```bash
# see registered model adapters
python -m jev.cli models

# inspect the dataset (sources, counts, types)
python -m jev.cli sources --data benchmark.json

# eyeball how an item is parsed
python -m jev.cli inspect --data benchmark.json --source winogrande --limit 2

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

`benchmark.json` is intentionally **not** committed (113 MB); pass `--data`.

## Scoring methods

Two strategies, selectable with `--score-mode` (each recipe picks a sensible
default, `--score-mode` overrides it globally):

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
