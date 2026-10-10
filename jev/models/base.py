"""Model adapter interface and registry.

Everything a model family must provide is described by :class:`ModelAdapter`.
The rest of the framework only ever talks to this interface, so adding support
for a new family means adding one file that registers an adapter - no existing
file has to change.

Registration
------------
::

    from jev.models.base import ModelAdapter, register_model

    @register_model("my-family")
    class MyAdapter(ModelAdapter):
        ...

Adapters are instantiated through :func:`build_model` / :func:`get_model`,
which look the name up in the registry populated by :mod:`jev.models.registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class ContinuationScore:
    """Result of scoring a continuation ``c`` given a context.

    ``logprob`` is the *sum* of per-token log-probabilities of the continuation
    tokens. ``num_tokens`` is how many tokens the continuation spans (useful for
    length normalisation). ``is_greedy`` reports whether the continuation is the
    model's own greedy continuation (diagnostic only).
    """

    logprob: float
    num_tokens: int
    is_greedy: bool = False

    @property
    def mean_logprob(self) -> float:
        return self.logprob / max(self.num_tokens, 1)


@dataclass
class ModelConfig:
    """Runtime configuration shared by all adapters."""

    model_id: Optional[str] = None          # local path or HF hub id
    device: Optional[str] = None            # "cuda", "cuda:0", "cpu", "auto"
    dtype: str = "auto"                     # "auto", "float16", "bfloat16", "float32"
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = True
    max_context_tokens: Optional[int] = None  # None -> infer from the model
    chat_template: Optional[str] = None       # override tokenizer chat template
    system_prompt: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Base class for all model adapters.

    Subclasses must implement :meth:`load`, :meth:`encode`, :meth:`decode`,
    :meth:`render_chat` and :meth:`loglikelihood`. A default
    :meth:`next_token_logprobs` implementation via :meth:`loglikelihood`-style
    forward pass is provided by concrete backends.
    """

    #: Registry name, set by the decorator.
    name: str = "base"

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()

    # -- lifecycle -------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load tokenizer/model weights into memory."""

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum number of tokens the model can attend over."""

    # -- tokenization ----------------------------------------------------
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Tokenize ``text`` into token ids."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Detokenize token ids."""

    @abstractmethod
    def render_chat(
        self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Render chat messages into a single prompt string."""

    # -- scoring ---------------------------------------------------------
    @abstractmethod
    def loglikelihood(self, context: str, continuation: str) -> ContinuationScore:
        """Sum of log-probabilities of ``continuation`` after ``context``.

        Only the tokens of ``continuation`` are scored; ``context`` tokens are
        conditioned on but not counted.
        """

    def loglikelihood_batch(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[ContinuationScore]:
        """Score many (context, continuation) pairs. Default: sequential loop."""
        return [self.loglikelihood(ctx, cont) for ctx, cont in pairs]

    def next_token_logprobs(
        self, context: str, token_ids: Sequence[Sequence[int]]
    ) -> List[float]:
        """Log-probabilities of candidate *first* tokens given ``context``.

        ``token_ids`` is a list of token-id sequences; only the first id of each
        is scored. Implemented by concrete backends.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement next_token_logprobs"
        )

    # -- helpers ---------------------------------------------------------
    def continuation_token_ids(self, context: str, continuation: str) -> List[int]:
        """Return the token ids of ``continuation`` when appended to ``context``.

        Handles tokenizer boundary effects by tokenizing ``context+continuation``
        and stripping the context prefix, so the returned ids exactly match what
        the model would see.
        """
        if not continuation:
            return []
        ctx_ids = self.encode(context, add_special_tokens=False)
        full_ids = self.encode(context + continuation, add_special_tokens=False)
        if full_ids[: len(ctx_ids)] == ctx_ids:
            return full_ids[len(ctx_ids):]
        # The tokenizer merged characters across the boundary; encode the
        # continuation on its own as a safe fallback.
        return self.encode(continuation, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FACTORIES: Dict[str, Callable[[Optional[ModelConfig]], ModelAdapter]] = {}
_CLASSES: Dict[str, type] = {}


def register_model(name: str):
    """Decorator registering a :class:`ModelAdapter` subclass under ``name``."""

    def deco(cls):
        if not issubclass(cls, ModelAdapter):
            raise TypeError("register_model expects a ModelAdapter subclass")
        cls.name = name
        _CLASSES[name] = cls
        _FACTORIES[name] = lambda config=None, _cls=cls: _cls(config)
        return cls

    return deco


def available_models() -> List[str]:
    return sorted(_CLASSES)


def get_model_class(name: str) -> type:
    if name not in _CLASSES:
        raise KeyError(
            f"unknown model adapter '{name}'. available: {available_models()}"
        )
    return _CLASSES[name]


def get_model(name: str, config: Optional[ModelConfig] = None) -> ModelAdapter:
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown model adapter '{name}'. available: {available_models()}"
        )
    return _FACTORIES[name](config)


def build_model(
    name: str, config: Optional[ModelConfig], load: bool = True
) -> ModelAdapter:
    """Instantiate an adapter and (optionally) load its weights."""
    adapter = get_model(name, config)
    if load:
        adapter.load()
    return adapter
