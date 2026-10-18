"""Turn intent-and-slot extraction into Jev selections and assemble the task JSON.

Jev returns typed answers, not text (design.md "Hosted model and Spend_Cap"):

* state: the request plus the 10 Train_Split exemplars with their gold JSON;
* one ``choice`` over every Train_Split intent;
* one ``choice`` per candidate span (every contiguous run of at most
  ``max_span_tokens`` whitespace tokens) over every Train_Split slot type plus
  ``none``.

:func:`decode` keeps each span whose most probable answer is a slot type,
greedily from the most probable down (ties: longer span, then earlier), skipping
spans that overlap one already kept, and lists the kept slots in request order.
Question wording comes from config/prompts/jev.yaml.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fts.data import Example, canonical_target
from fts.settings import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = REPO_ROOT / "config" / "prompts" / "jev.yaml"
INTENT_ID = "intent"
NONE = "none"


@dataclass(frozen=True)
class Span:
    start: int  # first token index
    end: int  # one past the last token index
    text: str

    @property
    def qid(self) -> str:
        return f"span_{self.start}_{self.end}"


@dataclass(frozen=True)
class JevTemplates:
    intent: str
    span: str
    none: str


def load_templates(path: Path | str = TEMPLATES) -> JevTemplates:
    d = load_yaml(path)
    return JevTemplates(intent=d["intent"].strip(), span=d["span"].strip(), none=d["none"].strip())


def candidate_spans(utterance: str, max_tokens: int) -> list[Span]:
    """Every contiguous span of 1..max_tokens whitespace tokens, in order of start then length."""
    tokens = utterance.split()
    return [Span(i, j, " ".join(tokens[i:j])) for i in range(len(tokens)) for j in range(i + 1, min(i + max_tokens, len(tokens)) + 1)]


def build_state(utterance: str, exemplars: Sequence[Example]) -> dict:
    return {"request": utterance, "examples": [{"request": e.inputs, "answer": json.loads(e.target)} for e in exemplars]}


def intent_question(t: JevTemplates, labels: dict) -> dict:
    return {"type": "choice", "instructions": t.intent, "criteria": {name: None for name in labels["intents"]}}


def span_question(t: JevTemplates, labels: dict, span: Span, occurrence: str = "") -> dict:
    criteria = {name: None for name in labels["slot_types"]}
    criteria[NONE] = t.none
    return {"type": "choice", "instructions": t.span.replace("{span}", span.text) + occurrence, "criteria": criteria}


def question_batches(utterance: str, labels: dict, t: JevTemplates, max_tokens: int, per_request: int) -> tuple[list[Span], list[dict]]:
    """The candidate spans and the question maps to send (the intent question rides in the first)."""
    spans = candidate_spans(utterance, max_tokens)
    counts: dict[str, int] = {}
    totals = {s.text: sum(1 for x in spans if x.text == s.text) for s in spans}
    questions = []
    for s in spans:
        counts[s.text] = counts.get(s.text, 0) + 1
        occurrence = f" (occurrence {counts[s.text]} of {totals[s.text]} of this text in `request`)" if totals[s.text] > 1 else ""
        questions.append((s.qid, span_question(t, labels, s, occurrence)))
    batches = [dict(questions[i : i + per_request]) for i in range(0, len(questions), per_request)] or [{}]
    batches[0] = {INTENT_ID: intent_question(t, labels), **batches[0]}
    return spans, batches


def decode(answers: dict, spans: Sequence[Span]) -> str:
    """The task JSON from Jev's answers (merged across requests)."""
    intent = answers[INTENT_ID]["choice"]
    candidates = []
    for s in spans:
        a = answers.get(s.qid)
        if a is None or a["choice"] == NONE:
            continue
        candidates.append((float(a["probabilities"].get(a["choice"], 0.0)), s, a["choice"]))
    candidates.sort(key=lambda c: (-c[0], -(c[1].end - c[1].start), c[1].start))
    taken: set[int] = set()
    kept = []
    for _, s, slot_type in candidates:
        positions = set(range(s.start, s.end))
        if positions & taken:
            continue
        taken |= positions
        kept.append((s, slot_type))
    kept.sort(key=lambda k: k[0].start)
    return canonical_target(intent, [(slot_type, s.text) for s, slot_type in kept])
