"""The generic recipe covering choice / noul / score.

It is registered at the lowest priority so any source-specific recipe overrides
it, while every source keeps working with no extra code.
"""

from __future__ import annotations

from typing import Optional

from ..data.schema import CHOICE, NOUL, SCORE, Candidate, Item
from ..models.base import ModelAdapter
from ..prompts.format import (
    build_context,
    fit_text,
    format_options,
    render_state,
    state_budget,
)
from ..prompts.options import PromptOptions
from .base import Recipe, RenderedQuery, register_recipe

ANSWER_PREFIX_HINTS = {
    CHOICE: "Answer with a single option.",
    NOUL: "Answer with a single word.",
    SCORE: "Answer with a single value from the scale.",
}


class GenericRecipe(Recipe):
    name = "generic"
    priority = -100

    def matches(self, item: Item) -> bool:  # fallback for everything
        return True

    # -- continuation strings -------------------------------------------
    def _continuation(self, item: Item, cand: Candidate, options: PromptOptions) -> str:
        if item.question_type == NOUL:
            base = options.noul_words[0] if cand.id == "true" else options.noul_words[1]
        else:
            base = cand.text
        if options.continuation_lead_space:
            return " " + base
        return base

    def _options_block(self, item: Item, options: PromptOptions) -> str:
        if item.question_type == CHOICE:
            return "Options:\n" + format_options(item.candidates, options.label_options)
        if item.question_type == NOUL:
            t, f = options.noul_words
            lines = [f"- {t}" + (f": {item.criteria[t]}" if isinstance(item.criteria, dict) and t in item.criteria else "") ]
            lines.append(f"- {f}" + (f": {item.criteria[f]}" if isinstance(item.criteria, dict) and f in item.criteria else ""))
            return "Answer options:\n" + "\n".join(lines)
        if item.question_type == SCORE:
            return "Answer options:\n" + "\n".join(
                f"- {c.text}" for c in item.candidates
            )
        return ""

    def build(
        self,
        item: Item,
        adapter: ModelAdapter,
        options: PromptOptions,
        mode: Optional[str] = None,
    ) -> RenderedQuery:
        score_mode = mode or "continuation"
        candidates = list(item.candidates)

        # Prepare continuation texts (used only in continuation mode).
        for c in candidates:
            c.token = c.token or c.text
        if score_mode == "continuation":
            cont_texts = [self._continuation(item, c, options) for c in candidates]
        else:
            cont_texts = [c.token for c in candidates]

        hint = ANSWER_PREFIX_HINTS.get(item.question_type, "")
        options_block = (self._options_block(item, options) + "\n" + hint).strip()

        budget = state_budget(adapter, options, item.instructions, options_block)
        state_text = fit_text(
            adapter, render_state(item.state), budget, options.state_fit
        )

        context = build_context(adapter, options, state_text, item.instructions, options_block)

        # Set the exact continuation string used for scoring (kept separate from
        # the human-readable option text).
        for c, txt in zip(candidates, cont_texts):
            c.continuation = txt

        return RenderedQuery(
            context=context,
            candidates=candidates,
            score_mode=score_mode,
            normalize="length",
            gold_id=item.gold_id,
            item_id=item.id,
            source=item.source,
            question_type=item.question_type,
            meta={"options_block": options_block},
        )


register_recipe(GenericRecipe())
