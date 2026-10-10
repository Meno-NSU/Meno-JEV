"""Generic HuggingFace causal-LM adapter.

This single adapter covers the large majority of decoder-only families
(Qwen, Llama, Mistral, Gemma, Phi, GPT-NeoX, ...). Family-specific quirks are
handled by subclasses in their own files; this file stays generic.

``torch`` and ``transformers`` are imported lazily so that the rest of the
framework (and the test suite) works on machines without them.

Memory
------
Scoring a candidate only needs the logits at the *last few* positions, not the
full ``[batch, seq_len, vocab]`` tensor. Computing the full tensor is what blows
up memory on long contexts (hundreds of candidates x long prompt x 150k vocab =
tens of GB). We therefore:

* ask the model for only the last ``max_continuation_len + 1`` logits via
  ``logits_to_keep`` (``num_logits_to_keep`` on older transformers);
* left-pad batches so those last positions stay aligned;
* batch adaptively by a token budget and retry with smaller sub-batches on OOM.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from .base import ContinuationScore, ModelAdapter, ModelConfig, register_model

# Quiet the CUDA caching-allocator OOM chatter (each failed allocation logs a
# glog "W..." warning). OOM events are still caught and summarised by us.
# Set before torch initialises CUDA; setdefault() respects user overrides.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _shim_torch_accelerator(torch) -> None:
    """Make ``torch.accelerator.current_accelerator`` return ``None`` instead of
    raising when no accelerator is present.

    Newer transformers call this at import time (indirectly, via
    ``get_device_type``); on a machine with no visible GPU it raises
    ``RuntimeError: Cannot access accelerator device when none is available``
    and breaks ``from transformers import AutoModelForCausalLM``. Returning
    ``None`` matches the intended "no accelerator" semantics and lets CPU runs
    work. A present GPU is unaffected (the original result is returned).
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return
    original = getattr(accelerator, "current_accelerator", None)
    if original is None or getattr(original, "_jev_safe", False):
        return

    def safe_current_accelerator(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return None

    safe_current_accelerator._jev_safe = True  # type: ignore[attr-defined]
    accelerator.current_accelerator = safe_current_accelerator


def _resolve_torch_dtype(torch, name: str):
    if name in (None, "auto"):
        return "auto"
    return {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }.get(name, "auto")


@register_model("hf")
class HFCausalAdapter(ModelAdapter):
    """Decoder-only causal LM scored through ``AutoModelForCausalLM``."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__(config)
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._logits_keep_arg: Optional[str] = None
        # None -> pick automatically from GPU memory after load().
        self._max_batch_tokens: Optional[int] = (
            int(self.config.extra["max_batch_tokens"])
            if self.config.extra.get("max_batch_tokens")
            else None
        )
        self.oom_retries = 0
        self._oom_notice_shown = False

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PyTorch is not installed in this environment. Install the [hf] "
                "extra (see SETUP.md step 3-4)."
            ) from exc

        # Guard against the transformers import crashing when no GPU is visible.
        _shim_torch_accelerator(torch)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import transformers (this happens before any Meno-JEV "
                "code runs, so it is an environment problem, not a Meno-JEV bug).\n"
                f"Underlying error: {type(exc).__name__}: {exc}\n"
                "Common causes and fixes:\n"
                "  * 'Cannot access accelerator device when none is available' -> "
                "this process sees no GPU. Check `nvidia-smi` and $CUDA_VISIBLE_DEVICES; "
                "if you meant to use the GPU, make it visible (e.g. "
                "CUDA_VISIBLE_DEVICES=0) or don't set it to an empty/invalid value. "
                "A torch/transformers version mismatch can also trigger this: "
                "`pip install -U torch transformers`.\n"
                "  * otherwise run `python -m jev.cli doctor` to inspect the env."
            ) from exc

        self._torch = torch
        cfg = self.config
        if not cfg.model_id:
            raise ValueError("ModelConfig.model_id is required")

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        if cfg.chat_template:
            self._tokenizer.chat_template = cfg.chat_template

        model_kwargs: Dict[str, object] = {
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": _resolve_torch_dtype(torch, cfg.dtype),
        }
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        device = cfg.device
        if device in (None, "auto"):
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            device = None
        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)

        if "device_map" not in model_kwargs:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(device)
            self._device = device
        else:
            self._device = getattr(self._model, "device", None)

        self._model.eval()
        self._detect_logits_keep_arg()
        if self._max_batch_tokens is None:
            self._max_batch_tokens = self._auto_max_batch_tokens()

    def _auto_max_batch_tokens(self) -> int:
        """Choose a per-forward token budget from the visible GPU memory.

        With the logits-truncation optimisation, peak memory is dominated by
        activations (roughly proportional to #tokens), not by the logits tensor,
        so a large-memory GPU can safely batch far more than the old 8192.
        """
        torch = self._torch
        try:
            if torch is not None and torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # ~512 tokens per GB of VRAM, clamped to a sane range.
                return int(max(8192, min(65536, total_gb * 512)))
        except Exception:  # noqa: BLE001
            pass
        return 8192

    @property
    def max_batch_tokens(self) -> int:
        return int(self._max_batch_tokens or 8192)

    def _detect_logits_keep_arg(self) -> None:
        """Work out how to ask the model for only the last few logits."""
        try:
            params = inspect.signature(self._model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                self._logits_keep_arg = name
                return
        self._logits_keep_arg = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def max_context_tokens(self) -> int:
        if self.config.max_context_tokens:
            return int(self.config.max_context_tokens)
        mconf = getattr(self._model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            val = getattr(mconf, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return 4096

    # -- tokenization ----------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(ids))

    def render_chat(
        self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        tok = self._tokenizer
        template_kwargs = dict(self.config.extra.get("chat_template_kwargs", {}) or {})
        if getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
            except TypeError:
                # Older templates don't accept the extra kwargs.
                return tok.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts) + (" " if add_generation_prompt else "")

    # -- internals -------------------------------------------------------
    def _forward_logits(self, input_ids, attention_mask, position_ids, keep: int):
        """Run the model, computing at most ``keep`` trailing logits.

        Returns a tensor ``[batch, block_len, vocab]`` whose positions are
        aligned to the *right* end of each sequence.
        """
        torch = self._torch
        kwargs: Dict[str, object] = {}
        if self._logits_keep_arg:
            kwargs[self._logits_keep_arg] = keep
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        return out.logits

    def _is_oom(self, exc: Exception) -> bool:
        torch = self._torch
        oom = getattr(torch.cuda, "OutOfMemoryError", None) if torch else None
        if oom is not None and isinstance(exc, oom):
            return True
        return "out of memory" in str(exc).lower()

    def _prep(self, context: str, continuation: str) -> Tuple[int, List[int], List[int]]:
        """Return (ctx_len, cont_ids, full_ids) with the context trimmed to fit."""
        max_len = self.max_context_tokens
        ctx_ids = self.encode(context, add_special_tokens=False)
        cont_ids = self.continuation_token_ids(context, continuation)
        if not cont_ids:
            return 0, [], []
        if len(cont_ids) > max_len:
            cont_ids = cont_ids[-max_len:]
        keep_ctx = max(0, max_len - len(cont_ids))
        ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx else []
        return len(ctx_ids), cont_ids, ctx_ids + cont_ids

    def _score_group(
        self,
        mem: List[Tuple[int, List[int], List[int]]],
        indices: Sequence[int],
        results: List[Optional[ContinuationScore]],
    ) -> None:
        torch = self._torch
        seqs = [mem[i] for i in indices]
        maxlen = max(len(s[2]) for s in seqs)
        keep = max((len(s[1]) for s in seqs), default=1) + 1

        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", None) or 0

        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for row, (_clen, _cids, full_ids) in enumerate(seqs):
            start = maxlen - len(full_ids)  # left padding
            input_ids[row, start:] = torch.tensor(full_ids, dtype=torch.long)
            attention[row, start:] = 1
        # Left padding: positions must be computed from the real start.
        position_ids = attention.cumsum(dim=1) - 1
        position_ids.clamp_(min=0)
        position_ids.masked_fill_(attention == 0, 0)

        if self._device is not None:
            input_ids = input_ids.to(self._device)
            attention = attention.to(self._device)
            position_ids = position_ids.to(self._device)

        logits = self._forward_logits(input_ids, attention, position_ids, keep)
        block_len = logits.shape[1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        for row, i in enumerate(indices):
            _clen, cont_ids, _full = seqs[row]
            n = len(cont_ids)
            # Predictors for the continuation sit in the last n positions of the
            # right-aligned block.
            rel = logprobs[row, block_len - n - 1: block_len - 1]
            total = 0.0
            is_greedy = True
            for j, tok in enumerate(cont_ids):
                step = rel[j]
                total += float(step[tok].item())
                if int(torch.argmax(step).item()) != tok:
                    is_greedy = False
            results[i] = ContinuationScore(total, n, is_greedy)

    def _score_recursive(
        self,
        mem: List[Tuple[int, List[int], List[int]]],
        indices: Sequence[int],
        results: List[Optional[ContinuationScore]],
    ) -> None:
        if not indices:
            return
        try:
            self._score_group(mem, indices, results)
        except Exception as exc:  # noqa: BLE001
            if not self._is_oom(exc):
                raise
            self.oom_retries += 1
            if not self._oom_notice_shown:
                self._oom_notice_shown = True
                print(
                    "[jev] CUDA OOM while scoring; shrinking batches automatically. "
                    "Lower --max-batch-tokens / --max-context-tokens to avoid it. "
                    "(further OOMs are summarised at the end)",
                    file=sys.stderr,
                )
            torch = self._torch
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(indices) == 1:
                raise
            # Permanently back off the budget so later items stop hitting OOM.
            self._max_batch_tokens = max(2048, self.max_batch_tokens // 2)
            mid = len(indices) // 2
            self._score_recursive(mem, indices[:mid], results)
            self._score_recursive(mem, indices[mid:], results)

    def _group_by_budget(self, mem: List[Tuple[int, List[int], List[int]]]) -> List[List[int]]:
        """Group sequence indices so each group fits the token budget."""
        groups: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0
        for i, (_clen, _cids, full_ids) in enumerate(mem):
            if not full_ids:
                continue
            length = len(full_ids)
            if current and current_tokens + length > self.max_batch_tokens:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(i)
            current_tokens += length
        if current:
            groups.append(current)
        return groups

    # -- main entry points ----------------------------------------------
    def loglikelihood_batch(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[ContinuationScore]:
        if not pairs:
            return []
        mem = [self._prep(ctx, cont) for ctx, cont in pairs]
        results: List[Optional[ContinuationScore]] = [None] * len(mem)
        for i, (_clen, cont_ids, _full) in enumerate(mem):
            if not cont_ids:
                results[i] = ContinuationScore(0.0, 0, True)
        for group in self._group_by_budget(mem):
            self._score_recursive(mem, group, results)
        return [r if r is not None else ContinuationScore(0.0, 0, True) for r in results]

    def loglikelihood(self, context: str, continuation: str) -> ContinuationScore:
        return self.loglikelihood_batch([(context, continuation)])[0]

    # -- next-token mode -------------------------------------------------
    def next_token_logprobs(
        self, context: str, token_ids: Sequence[Sequence[int]]
    ) -> List[float]:
        torch = self._torch
        ctx_ids = self.encode(context, add_special_tokens=False)
        max_len = self.max_context_tokens
        if len(ctx_ids) > max_len:
            ctx_ids = ctx_ids[-max_len:]
        input_ids = torch.tensor([ctx_ids], dtype=torch.long)
        attention = torch.ones_like(input_ids)
        position_ids = torch.arange(len(ctx_ids), dtype=torch.long).unsqueeze(0)
        if self._device is not None:
            input_ids = input_ids.to(self._device)
            attention = attention.to(self._device)
            position_ids = position_ids.to(self._device)
        logits = self._forward_logits(input_ids, attention, position_ids, keep=1)
        step = torch.log_softmax(logits[0, -1].float(), dim=-1)
        out: List[float] = []
        for ids in token_ids:
            out.append(float(step[ids[0]].item()) if ids else float("-inf"))
        return out
