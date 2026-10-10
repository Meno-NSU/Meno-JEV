"""Generic HuggingFace causal-LM adapter.

This single adapter covers the large majority of decoder-only families
(Qwen, Llama, Mistral, Gemma, Phi, GPT-NeoX, ...). Family-specific quirks are
handled by subclasses in their own files; this file stays generic.

``torch`` and ``transformers`` are imported lazily so that the rest of the
framework (and the test suite) works on machines without them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .base import ContinuationScore, ModelAdapter, ModelConfig, register_model


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

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Fallback for tokenizers without a chat template.
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
    def _prepare_inputs(self, full_ids: List[int]):
        """Truncate from the left if needed and move to a torch tensor."""
        torch = self._torch
        max_len = self.max_context_tokens
        if len(full_ids) > max_len:
            full_ids = full_ids[-max_len:]
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        if self._device is not None:
            input_ids = input_ids.to(self._device)
        return input_ids

    def loglikelihood(self, context: str, continuation: str) -> ContinuationScore:
        torch = self._torch
        ctx_ids = self.encode(context, add_special_tokens=False)
        cont_ids = self.continuation_token_ids(context, continuation)
        if not cont_ids:
            return ContinuationScore(0.0, 0, True)

        full_ids = ctx_ids + cont_ids
        max_len = self.max_context_tokens
        if len(full_ids) > max_len:
            # keep the continuation intact, drop context from the left
            keep_ctx = max_len - len(cont_ids)
            if keep_ctx <= 0:
                cont_ids = cont_ids[-max_len:]
                full_ids = cont_ids
                ctx_len = 0
            else:
                ctx_ids = ctx_ids[-keep_ctx:]
                full_ids = ctx_ids + cont_ids
                ctx_len = len(ctx_ids)
        else:
            ctx_len = len(ctx_ids)

        input_ids = self._prepare_inputs(full_ids)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids).logits[0]

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        is_greedy = True
        n_full = len(full_ids)
        offset = n_full - len(cont_ids)
        for j, tok in enumerate(cont_ids):
            pos = offset + j
            if pos < 1:
                continue
            step = logprobs[pos - 1]
            total += float(step[tok].item())
            if int(torch.argmax(step).item()) != tok:
                is_greedy = False
        return ContinuationScore(total, len(cont_ids), is_greedy)

    # -- batched scoring -------------------------------------------------
    def loglikelihood_batch(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[ContinuationScore]:
        if not pairs:
            return []
        torch = self._torch
        max_len = self.max_context_tokens
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", None) or 0

        seqs = []  # (ctx_len, cont_ids, full_ids)
        for context, continuation in pairs:
            ctx_ids = self.encode(context, add_special_tokens=False)
            cont_ids = self.continuation_token_ids(context, continuation)
            if not cont_ids:
                seqs.append((0, [], []))
                continue
            if len(ctx_ids) + len(cont_ids) > max_len:
                keep_ctx = max(0, max_len - len(cont_ids))
                ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx else []
                if len(cont_ids) > max_len:
                    cont_ids = cont_ids[-max_len:]
            full_ids = ctx_ids + cont_ids
            seqs.append((len(ctx_ids), cont_ids, full_ids))

        longest = max((len(s[2]) for s in seqs), default=1) or 1
        batch = len(seqs)
        input_ids = torch.full((batch, longest), pad_id, dtype=torch.long)
        attention = torch.zeros((batch, longest), dtype=torch.long)
        for i, (_clen, _cids, full_ids) in enumerate(seqs):
            if not full_ids:
                continue
            input_ids[i, : len(full_ids)] = torch.tensor(full_ids, dtype=torch.long)
            attention[i, : len(full_ids)] = 1
        if self._device is not None:
            input_ids = input_ids.to(self._device)
            attention = attention.to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention).logits

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        results: List[ContinuationScore] = []
        for i, (_clen, cont_ids, full_ids) in enumerate(seqs):
            if not cont_ids:
                results.append(ContinuationScore(0.0, 0, True))
                continue
            offset = len(full_ids) - len(cont_ids)
            total = 0.0
            is_greedy = True
            for j, tok in enumerate(cont_ids):
                pos = offset + j
                if pos < 1:
                    continue
                step = logprobs[i, pos - 1]
                total += float(step[tok].item())
                if int(torch.argmax(step).item()) != tok:
                    is_greedy = False
            results.append(ContinuationScore(total, len(cont_ids), is_greedy))
        return results

    # -- next-token mode -------------------------------------------------
    def next_token_logprobs(
        self, context: str, token_ids: Sequence[Sequence[int]]
    ) -> List[float]:
        torch = self._torch
        ctx_ids = self.encode(context, add_special_tokens=False)
        input_ids = self._prepare_inputs(ctx_ids)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out: List[float] = []
        for ids in token_ids:
            if not ids:
                out.append(float("-inf"))
            else:
                out.append(float(logprobs[ids[0]].item()))
        return out
