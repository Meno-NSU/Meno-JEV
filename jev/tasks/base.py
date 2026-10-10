"""Task recipes: turn a parsed item into a scorable query.

A :class:`Recipe` decides the prompt layout and the scoring mode for a set of
items. The generic recipe handles all three question types out of the box;
source-specific behaviour is added by *registering* another recipe with a
higher priority - existing recipes are never edited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..data.schema import Item
from ..models.base import ModelAdapter
from ..prompts.options import PromptOptions


@dataclass
class RenderedQuery:
    context: str
    candidates: List  # List[Candidate]
    score_mode: str = "continuation"
    normalize: str = "length"
    temperature: float = 1.0
    gold_id: str = ""
    item_id: str = ""
    source: str = ""
    question_type: str = ""
    meta: Dict[str, object] = field(default_factory=dict)


class Recipe(ABC):
    name: str = "recipe"
    priority: int = 0

    @abstractmethod
    def matches(self, item: Item) -> bool:
        ...

    @abstractmethod
    def build(
        self,
        item: Item,
        adapter: ModelAdapter,
        options: PromptOptions,
        mode: Optional[str] = None,
    ) -> RenderedQuery:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# (priority, recipe); higher priority wins, generic recipes are negative.
_REGISTRY: List[Tuple[int, Recipe]] = []


def register_recipe(recipe: Recipe, priority: Optional[int] = None) -> Recipe:
    prio = recipe.priority if priority is None else priority
    recipe.priority = prio
    _REGISTRY.append((prio, recipe))
    _REGISTRY.sort(key=lambda t: t[0], reverse=True)
    return recipe


def resolve_recipe(item: Item) -> Recipe:
    for _prio, recipe in _REGISTRY:
        if recipe.matches(item):
            return recipe
    raise LookupError(f"no recipe matched item {item.id}")


def iter_recipes() -> List[Tuple[int, Recipe]]:
    return list(_REGISTRY)
