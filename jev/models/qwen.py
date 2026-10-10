"""Qwen family specialization.

Qwen2 / Qwen2.5 / Qwen3 tokenizers ship a ChatML chat template already, so the
generic HF adapter works as-is. This adapter only applies the small fixes that
the family actually needs and gives a stable name ("qwen") to select.

Add another family by copying this file; do not edit it.
"""

from __future__ import annotations

from .base import ModelConfig, register_model
from .hf_causal import HFCausalAdapter

# Minimal ChatML fallback if a tokenizer somehow lacks a template.
_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


@register_model("qwen")
class QwenAdapter(HFCausalAdapter):
    """HuggingFace causal adapter with Qwen-specific defaults."""

    def load(self) -> None:
        super().load()
        tok = self._tokenizer
        # Qwen tokenizers normally have no pad token; batching needs one.
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        # If the fine-tune stripped the template, restore a ChatML default.
        if not getattr(tok, "chat_template", None) and not self.config.chat_template:
            tok.chat_template = _QWEN_CHAT_TEMPLATE
