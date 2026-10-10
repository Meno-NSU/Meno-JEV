# Meno-JEV — environment setup

Meno-JEV has **two install profiles**:

| profile | what you get | needs GPU? | uses |
| --- | --- | --- | --- |
| **core** | `menoj` + `tqdm` | no | `smoke`, `sources`, `inspect`, tests, the `mock` adapter |
| **hf** | core + `torch`/`transformers`/`accelerate` | optional | real models (`--adapter qwen`/`hf`) |

The framework imports `torch`/`transformers` **lazily**, so the core profile
works on any machine; the real model is only touched when you run with a real
adapter.

---

## 0. Prerequisites

- **Python 3.9+**. Recommended **3.10 or 3.11**. Avoid brand-new interpreters
  (e.g. 3.13/3.14) — PyTorch wheels may not exist for them yet.
- `git`.
- For real models: an NVIDIA driver (and CUDA toolkit) or a CPU; HuggingFace
  models are downloaded on first use.

Check your interpreter:

```bash
python3 --version
```

If it is too new, point the setup at a specific binary: `PYTHON=python3.11 bash scripts/setup_env.sh`.

## 1. Get the code

```bash
git clone git@github.com:Meno-NSU/Meno-JEV.git
cd Meno-JEV
```

> If port 22 is blocked (common on clusters), use SSH over 443 or HTTPS:
> ```bash
> git clone ssh://git@ssh.github.com:443/Meno-NSU/Meno-JEV.git
> # or
> git clone https://github.com/Meno-NSU/Meno-JEV.git
> ```

Put `benchmark.json` in the repo root (it is **gitignored**, 113 MB — fetch it
separately, do not commit it). Every command takes `--data <path>`.

## 2. Create the environment

Two options — pick one.

### Option A: venv (fastest)

```bash
bash scripts/setup_env.sh                 # core
WITH_HF=1 bash scripts/setup_env.sh       # core + HuggingFace backend
source .venv/bin/activate
```

### Option B: conda

```bash
conda create -n menoj python=3.11 -y
conda activate menoj
pip install -e .            # core
```

## 3. Install PyTorch (real models only)

Install the torch build that matches your CUDA, **before** `transformers`:

```bash
# pick ONE:
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

Check the CUDA version your driver supports with `nvidia-smi` (top-right corner,
"CUDA Version"). If `pip install torch` with no index-url works for you, that is
fine too.

## 4. Install the rest of the backend

```bash
pip install -r requirements-hf.txt
# or equivalently: pip install -e ".[hf]"
```

## 5. Verify the environment

```bash
python -m jev.cli doctor
```

Example (core-only machine):

```
Meno-JEV environment doctor
  python: 3.11.9  (Linux x86_64)
  torch: NOT INSTALLED (ModuleNotFoundError)
  transformers: NOT INSTALLED (ModuleNotFoundError)
  cuda available: unknown (torch not installed)
  model adapters: hf, mock, qwen
  status: core OK; install the [hf] extra to run real models
```

On a GPU box you should see `cuda available: True`, the CUDA version and GPU
names. `doctor` is the fastest way to debug "why doesn't my model run".

## 6. Smoke-test the harness (no model, no GPU)

```bash
python -m jev.cli smoke --data benchmark.json
# -> checks: 464  failed: 0 ... RESULT: OK   (exit 0)
```

## 7. Run a real model

Local checkpoint:

```bash
python -m jev.cli run --data benchmark.json \
    --adapter qwen --model-id /path/to/your-finetune \
    --score-mode continuation --normalize length \
    --output-dir outputs --run-name tune_v1
```

HuggingFace Hub id (private/gated models need a token):

```bash
export HF_TOKEN=hf_xxx
python -m jev.cli run --data benchmark.json --adapter qwen --model-id Qwen/Qwen2.5-7B-Instruct
```

First quickly smoke the real model to confirm it loads and scores:

```bash
python -m jev.cli smoke --data benchmark.json --per-source-limit 1 \
    --adapter qwen --model-id /path/to/your-finetune
```

### Useful run flags

| flag | purpose |
| --- | --- |
| `--adapter qwen` / `hf` | which model family adapter to use |
| `--model-id PATH_OR_REPO` | local dir or HF hub id |
| `--dtype bfloat16` | precision (`auto`, `float16`, `bfloat16`, `float32`) |
| `--device cuda` / `cpu` | force device (`auto` = all GPUs via `device_map`) |
| `--attn sdpa` / `flash_attention_2` | attention implementation |
| `--max-context-tokens N` | cap context for long states (see below) |
| `--max-batch-tokens N` | token budget per forward batch (default 8192; lower it if you OOM) |
| `--score-mode continuation` / `next_token` | scoring method |
| `--normalize length` / `sum` / `none` | candidate log-prob normalisation |
| `--state-fit head` / `tail` / `head_tail` / `none` | how long states are truncated |
| `--sources a,b,c` `--types choice` `--split test` | filter the dataset |

## Environment variables

| variable | meaning |
| --- | --- |
| `HF_HOME` | where HuggingFace caches weights/tokenizers (set it to a big disk) |
| `HF_TOKEN` | auth for gated/private models |
| `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | never hit the network (use a pre-downloaded cache) |
| `CUDA_VISIBLE_DEVICES=0,1` | restrict which GPUs are used |
| `TOKENIZERS_PARALLELISM=false` | silences tokenizer fork warnings in multi-proc |

Pre-downloading a model for an offline cluster:

```bash
export HF_HOME=/scratch/$USER/hf
huggingface-cli download Qwen/Qwen2.5-7B-Instruct   # or: python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## What lives where (and what is not in git)

- Tracked: all code, tests, `SETUP.md`, `README.md`, `scripts/`, `requirements*.txt`.
- **Not tracked**: `benchmark.json` (113 MB), `outputs/` (predictions, `*.report.*`),
  `__pycache__`, `.venv`. Provide the data path with `--data`; results are written
  under `--output-dir`.

## Troubleshooting

- **`No module named 'transformers'` / `torch`** on a real run → you installed
  the core profile only. Do step 3–4, then `python -m jev.cli doctor`.
- **`git@github.com: ... Connection closed ... port 22`** → port 22 blocked; use
  `ssh://git@ssh.github.com:443/Meno-NSU/Meno-JEV.git` or HTTPS.
- **CUDA out of memory** → smaller `--dtype bfloat16`, lower
  `--max-context-tokens` and `--max-batch-tokens`, one GPU via
  `CUDA_VISIBLE_DEVICES=0`, or a smaller model. The adapter only computes the
  logits it needs and retries smaller sub-batches automatically, so OOM now
  usually means a single very long sequence is too big for one forward.
- **`causal_conv1d` / `flash-linear-attention` fallback warnings** on hybrid
  models (e.g. Qwen3.5) are harmless — they only mean a slower reference kernel
  is used. Install those packages for speed if desired.
- **`CUDA error: unspecified launch failure` / `canUse32BitIndexMath`** →
  usually a cascade after a prior OOM; fix memory first.
- **Very long states** (some up to ~328k chars) → set `--max-context-tokens` and
  `--state-fit head_tail`; long states are cut into the context.
- **`sentencepiece` build error** → `pip install sentencepiece` (some tokenizers
  need it); it is in `requirements-hf.txt`.
- **Gated model 401** → `export HF_TOKEN=...`.
```
