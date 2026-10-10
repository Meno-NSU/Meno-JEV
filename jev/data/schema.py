"""Parsing of the JEVal benchmark structure into a uniform item.

Each raw record in ``benchmark.json`` looks like::

    {
      "id": "jeval:winogrande:000199",
      "source": "winogrande",
      "state": "<context string, sometimes a JSON blob>",
      "question_type": "choice" | "noul" | "score",
      "questions_json": "<json string: {key: {type, instructions, criteria}}>",
      "label_json": "<json string: gold label>",
      "source_item_id": "winogrande/dev/1259"
    }

Every record holds exactly one sub-question. The three question types map to a
uniform notion of a set of :class:`Candidate`\\ s plus one gold candidate:

* ``choice`` - ``criteria`` is a dict ``{key: text}``; the gold label is a key.
* ``noul``   - binary; gold label is ``1`` (true) or ``0`` (false).
* ``score``  - ``criteria`` is a list; gold label is a 0-based index into it.

The parser never assumes ordering of criteria keys and always produces a stable,
ordered candidate list plus a canonical string id for the gold candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

CHOICE = "choice"
NOUL = "noul"
SCORE = "score"

QUESTION_TYPES = (CHOICE, NOUL, SCORE)


@dataclass
class Candidate:
    """One possible answer.

    Attributes
    ----------
    id:
        Canonical, stable identifier used for scoring/aggregation.
    text:
        Human-readable form, used when scoring a full-text continuation.
    token:
        Short single-token form (letter/digit/key), used by next-token scoring.
    raw:
        The original value from the benchmark (may be ``None``).
    """

    id: str
    text: str
    token: str = ""
    raw: Any = None
    continuation: str = ""

    def __post_init__(self) -> None:
        if not self.token:
            self.token = self.text
        if not self.continuation:
            self.continuation = self.text


@dataclass
class Item:
    id: str
    source: str
    state: str
    question_type: str
    source_item_id: str

    question_key: str = "q"
    instructions: str = ""
    criteria: Any = None

    gold_raw: Any = None
    gold_id: str = ""

    candidates: List[Candidate] = field(default_factory=list)

    split: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -----------------------------------------------------
    @property
    def state_is_json(self) -> bool:
        s = self.state.lstrip()
        return s.startswith("{") or s.startswith("[")

    def state_as_json(self) -> Optional[Any]:
        try:
            return json.loads(self.state)
        except (ValueError, TypeError):
            return None

    def candidate_by_id(self, cid: str) -> Optional[Candidate]:
        for c in self.candidates:
            if c.id == cid:
                return c
        return None

    def gold_candidate(self) -> Optional[Candidate]:
        return self.candidate_by_id(self.gold_id)


def _split_of(source_item_id: str) -> str:
    parts = source_item_id.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return ""


def _first_question(questions: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not questions:
        return "q", {}
    key = next(iter(questions))
    val = questions[key]
    if not isinstance(val, dict):
        return key, {}
    return key, val


def _letter_token(index: int) -> str:
    # 0->A, 25->Z, 26->AA ...
    name = ""
    n = index
    while True:
        name = chr(ord("A") + (n % 26)) + name
        n = n // 26 - 1
        if n < 0:
            break
    return name


def _build_choice_candidates(criteria: Dict[str, Any]) -> List[Candidate]:
    cands: List[Candidate] = []
    for i, (key, val) in enumerate(criteria.items()):
        text = key if val is None else str(val)
        # A per-option letter is only meaningful when the option is labeled by
        # an ``opt_k`` key; for named keys (entity/intent) the key is the token.
        token = _letter_token(i) if str(key).startswith("opt") else key
        cands.append(Candidate(id=str(key), text=text, token=str(token), raw=val))
    return cands


def _build_noul_candidates(criteria: Any) -> List[Candidate]:
    crit = criteria if isinstance(criteria, dict) else {}
    out = []
    for cid in ("true", "false"):
        text = crit.get(cid) if isinstance(crit, dict) else None
        if text is None:
            text = cid
        # Token for next-token scoring should be the literal boolean word.
        out.append(Candidate(id=cid, text=cid, token=cid, raw=crit.get(cid)))
    return out


def _build_score_candidates(criteria: Any) -> List[Candidate]:
    out: List[Candidate] = []
    values = criteria if isinstance(criteria, list) else []
    for i, val in enumerate(values):
        out.append(Candidate(id=str(i), text=str(val), token=str(val), raw=val))
    return out


def _resolve_gold(question_type: str, label: Any, candidates: List[Candidate]) -> str:
    if question_type == CHOICE:
        return str(label)
    if question_type == NOUL:
        # 1 -> true, 0 -> false. Accept bools / strings too.
        if isinstance(label, bool):
            return "true" if label else "false"
        if isinstance(label, (int, float)):
            return "true" if int(label) == 1 else "false"
        return str(label).strip().lower()
    if question_type == SCORE:
        return str(int(label))
    return str(label)


def parse_item(record: Dict[str, Any]) -> Item:
    """Turn one raw benchmark record into a parsed :class:`Item`."""
    qtype = record["question_type"]
    try:
        questions = json.loads(record.get("questions_json") or "{}")
    except (ValueError, TypeError):
        questions = {}
    key, qobj = _first_question(questions)

    try:
        label = json.loads(record.get("label_json"))
    except (ValueError, TypeError):
        label = record.get("label_json")

    criteria = qobj.get("criteria")
    instructions = qobj.get("instructions", "") or ""

    if qtype == CHOICE:
        cands = _build_choice_candidates(criteria or {})
    elif qtype == NOUL:
        cands = _build_noul_candidates(criteria)
    elif qtype == SCORE:
        cands = _build_score_candidates(criteria)
    else:
        cands = []

    item = Item(
        id=record["id"],
        source=record["source"],
        state=record.get("state", "") or "",
        question_type=qtype,
        source_item_id=record.get("source_item_id", ""),
        question_key=key,
        instructions=instructions,
        criteria=criteria,
        gold_raw=label,
        candidates=cands,
        split=_split_of(record.get("source_item_id", "")),
    )
    item.gold_id = _resolve_gold(qtype, label, cands)
    return item
