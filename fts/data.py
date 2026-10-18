"""Deterministic MASSIVE data pipeline: transform, official splits, overlap check (Req 2.3-2.7).

Each MASSIVE 1.1 record has ``id``, ``partition`` (``train``, ``dev`` or ``test``),
``intent`` and ``utt``, plus ``annot_utt`` in which slots are marked
``[label : entity]``. :func:`transform_example` turns one record into an
:class:`Example` whose target is the canonical JSON

    {"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]}

serialized with ``json.dumps(obj, ensure_ascii=False)``, so the same raw record
always yields the same string. A record is excluded (``None``, with the reason
available from :func:`transform_with_reason`) when a required field is missing,
the intent is empty, the brackets do not balance, a span lacks ``" : "``, or
removing the markup does not reproduce ``utt`` after collapsing whitespace.

Splits are MASSIVE's official partitions (``train`` -> Train, ``dev`` ->
Validation, ``test`` -> Test), so membership does not depend on the seed; the
seed is accepted and recorded for uniformity. No step here uses randomness.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PARTITION_TO_SPLIT = {"train": "train", "dev": "validation", "test": "test"}
SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_FIELDS = ("id", "partition", "intent", "utt", "annot_utt")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Example:
    example_id: str  # MASSIVE id; stable identifier for overlap checks and reporting
    partition: str  # "train" | "dev" | "test" (official MASSIVE partition)
    inputs: str  # the utterance (utt)
    target: str  # canonical gold JSON string


@dataclass(frozen=True)
class Splits:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]

    def by_name(self, name: str) -> tuple[Example, ...]:
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]


@dataclass(frozen=True)
class Exclusion:
    example_id: str
    reason: str


@dataclass
class TransformReport:
    examples: list[Example] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)

    @property
    def excluded_count(self) -> int:
        return len(self.exclusions)


class MalformedRecord(ValueError):
    """Raised by :func:`parse_annotation` with the reason a record is excluded."""


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def parse_annotation(annot_utt: str) -> tuple[list[tuple[str, str]], str]:
    """Return the ``(label, value)`` spans in order and the text with the markup removed."""
    slots: list[tuple[str, str]] = []
    plain: list[str] = []
    i = 0
    while i < len(annot_utt):
        ch = annot_utt[i]
        if ch == "[":
            end = annot_utt.find("]", i + 1)
            if end == -1 or "[" in annot_utt[i + 1 : end]:
                raise MalformedRecord("unbalanced brackets")
            inner = annot_utt[i + 1 : end]
            if " : " not in inner:
                raise MalformedRecord("slot span lacks ' : '")
            label, value = (part.strip() for part in inner.split(" : ", 1))
            if not label or not value:
                raise MalformedRecord("empty slot label or value")
            slots.append((label, value))
            plain.append(value)
            i = end + 1
        elif ch == "]":
            raise MalformedRecord("unbalanced brackets")
        else:
            plain.append(ch)
            i += 1
    return slots, collapse_ws("".join(plain))


def canonical_target(intent: str, slots: Sequence[tuple[str, str]]) -> str:
    obj = {"intent": intent, "slots": [{"type": t, "value": v} for t, v in slots]}
    return json.dumps(obj, ensure_ascii=False)


def transform_with_reason(raw: object) -> tuple[Example | None, str | None]:
    """The Example, or None with the reason the record is excluded."""
    if not isinstance(raw, dict):
        return None, "record is not a JSON object"
    for name in REQUIRED_FIELDS:
        if not isinstance(raw.get(name), str):
            return None, f"missing field {name}"
    intent = raw["intent"].strip()
    if not intent:
        return None, "empty intent"
    if raw["partition"] not in PARTITION_TO_SPLIT:
        return None, f"unknown partition {raw['partition']!r}"
    try:
        slots, plain = parse_annotation(raw["annot_utt"])
    except MalformedRecord as exc:
        return None, str(exc)
    utt = collapse_ws(raw["utt"])
    if plain != utt:
        return None, "annot_utt without markup does not reproduce utt"
    return Example(example_id=raw["id"], partition=raw["partition"], inputs=utt, target=canonical_target(intent, slots)), None


def transform_example(raw: dict, seed: int) -> Example | None:
    """Deterministic transform of one MASSIVE record; None for a malformed one (Req 2.5, 2.6).

    ``seed`` is part of the documented interface; the transform uses no randomness.
    """
    del seed
    return transform_with_reason(raw)[0]


def transform_all(raws: Iterable[object], seed: int) -> TransformReport:
    """Transform every record, collecting the examples and each exclusion's id and reason."""
    del seed
    report = TransformReport()
    for index, raw in enumerate(raws):
        example, reason = transform_with_reason(raw)
        if example is None:
            rid = raw.get("id") if isinstance(raw, dict) else None
            report.exclusions.append(Exclusion(rid if isinstance(rid, str) and rid else f"<line {index + 1}>", reason or "malformed"))
        else:
            report.examples.append(example)
    return report


def build_splits(examples: Sequence[Example], seed: int) -> Splits:
    """Assign each Example to the split named by its official partition, keeping dataset order (Req 2.3)."""
    del seed
    buckets: dict[str, list[Example]] = {name: [] for name in SPLIT_NAMES}
    for ex in examples:
        if ex.partition not in PARTITION_TO_SPLIT:
            raise ValueError(f"example {ex.example_id} has unknown partition {ex.partition!r}")
        buckets[PARTITION_TO_SPLIT[ex.partition]].append(ex)
    return Splits(train=tuple(buckets["train"]), validation=tuple(buckets["validation"]), test=tuple(buckets["test"]))


def _pairs(splits: Splits) -> list[tuple[str, tuple[Example, ...], tuple[Example, ...]]]:
    return [
        ("train-validation", splits.train, splits.validation),
        ("train-test", splits.train, splits.test),
        ("validation-test", splits.validation, splits.test),
    ]


def check_overlap(splits: Splits) -> dict[str, list[str]]:
    """{pair: sorted overlapping example ids} for every split pair that shares an id; empty when disjoint."""
    overlaps = {}
    for name, a, b in _pairs(splits):
        shared = {e.example_id for e in a} & {e.example_id for e in b}
        if shared:
            overlaps[name] = sorted(shared)
    return overlaps


def overlap_total(overlaps: dict[str, list[str]]) -> int:
    """Number of distinct example ids that appear in more than one split."""
    return len({i for ids in overlaps.values() for i in ids})


def shared_utterances(splits: Splits) -> dict[str, int]:
    """Count of distinct utterance texts shared by each split pair (reported, not failed)."""
    return {name: len({e.inputs for e in a} & {e.inputs for e in b}) for name, a, b in _pairs(splits)}


def split_counts(splits: Splits) -> dict[str, int]:
    return {name: len(splits.by_name(name)) for name in SPLIT_NAMES}


def labels_from_train(train: Sequence[Example]) -> dict[str, list[str]]:
    """Sorted intent and slot-type lists built from the Train_Split only."""
    intents, slot_types = set(), set()
    for ex in train:
        obj = json.loads(ex.target)
        intents.add(obj["intent"])
        slot_types.update(s["type"] for s in obj["slots"])
    return {"intents": sorted(intents), "slot_types": sorted(slot_types)}


def write_splits(splits: Splits, directory: Path | str) -> dict[str, Path]:
    """Write ``{train,validation,test}.jsonl`` with one Example per line."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SPLIT_NAMES:
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for ex in splits.by_name(name):
                row = {"id": ex.example_id, "partition": ex.partition, "utt": ex.inputs, "target": ex.target}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[name] = p
    return paths


def read_split(path: Path | str) -> tuple[Example, ...]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out.append(Example(example_id=row["id"], partition=row["partition"], inputs=row["utt"], target=row["target"]))
    return tuple(out)


def load_splits(directory: Path | str) -> Splits:
    d = Path(directory)
    return Splits(*(read_split(d / f"{name}.jsonl") for name in SPLIT_NAMES))
