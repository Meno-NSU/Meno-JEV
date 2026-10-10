"""TEMPLATE - copy this file to add a per-source recipe override.

This file is *not* imported (name starts with ``_``). To add an override:

1. Copy it to ``sources/<your_source>.py`` (no leading underscore).
2. Adjust ``matches`` and ``build``.
3. That's it - the package auto-imports the file and the recipe registers
   itself. Nothing else in the framework changes.

Recipes with a higher ``priority`` than the generic recipe (-100) win for the
items they match.
"""

from __future__ import annotations

from typing import Optional

from ...data.schema import Item
from ...models.base import ModelAdapter
from ...prompts.format import build_context, fit_text, format_options, render_state, state_budget
from ...prompts.options import PromptOptions
from ..base import Recipe, RenderedQuery, register_recipe


class MySourceRecipe(Recipe):
    name = "my-source"
    priority = 10

    def matches(self, item: Item) -> bool:
        return item.source == "REPLACE_ME"

    def build(
        self,
        item: Item,
        adapter: ModelAdapter,
        options: PromptOptions,
        mode: Optional[str] = None,
    ) -> RenderedQuery:
        score_mode = mode or "continuation"
        options_block = "Options:\n" + format_options(item.candidates, options.label_options)
        budget = state_budget(adapter, options, item.instructions, options_block)
        state_text = fit_text(adapter, render_state(item.state), budget, options.state_fit)
        context = build_context(adapter, options, state_text, item.instructions, options_block)
        return RenderedQuery(
            context=context,
            candidates=list(item.candidates),
            score_mode=score_mode,
            gold_id=item.gold_id,
            item_id=item.id,
            source=item.source,
            question_type=item.question_type,
        )


# register_recipe(MySourceRecipe())
