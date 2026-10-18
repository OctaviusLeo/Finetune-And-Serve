"""Build the MASSIVE 1.1 English dataset and record its provenance (task 2.4; Req 2.2-2.7, 0.8).

Steps: download the tarball into runs/cache/ (once), record its SHA-256, URL and
access date; read the ``en-US.jsonl`` member straight from the archive (nothing
else is extracted); transform every record; build the official splits and
overlap-check them (exit non-zero, listing every id, on any overlap); report the
utterance texts shared across splits; write data/massive_en/*.jsonl,
config/labels.json (Train_Split only) and the stamped results/data_counts.json;
measure the longest tokenized training example under the fine-tuning prompt
against ``max_seq_len``; and record MASSIVE and SLURP attribution in THIRD_PARTY.md.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tarfile
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from fts import data, envdoc, prompts, repro, settings

REPO_ROOT = Path(__file__).resolve().parent.parent
URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MEMBER_SUFFIX = "/en-US.jsonl"
CACHE = REPO_ROOT / "runs" / "cache" / "amazon-massive-dataset-1.1.tar.gz"
DATA_DIR = REPO_ROOT / "data" / "massive_en"
LABELS_JSON = REPO_ROOT / "config" / "labels.json"
COUNTS_JSON = REPO_ROOT / "results" / "data_counts.json"
THIRD_PARTY_MD = REPO_ROOT / "THIRD_PARTY.md"
SEED = 0

DATASET_SECTION = """| Name | Version | License | Source |
| --- | --- | --- | --- |
| MASSIVE (English, `en-US.jsonl`) | 1.1 | CC-BY-4.0 | {url} (SHA-256 `{sha256}`, accessed {accessed}) |
| SLURP (the English text MASSIVE derives from) | as included in MASSIVE 1.1 | CC-BY-4.0 | https://github.com/pswietojanski/slurp |

Attribution: MASSIVE, FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages" (2022), Amazon, licensed CC BY 4.0 (https://github.com/alexa/massive, `NOTICE.md`). SLURP, Bastianelli et al., "SLURP: A Spoken Language Understanding Resource Package" (2020), licensed CC BY 4.0. This repository commits a transformed copy of the English records (`data/massive_en/`): the utterance, the official partition, and the intent and slots rewritten as one JSON object."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str = URL, dest: Path = CACHE, opener: Callable = urllib.request.urlopen) -> Path:
    """Download ``url`` to ``dest`` unless it is already there (written atomically via a .part file)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with opener(url) as resp, tmp.open("wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)
    tmp.replace(dest)
    return dest


def iter_member_records(tar_path: Path, suffix: str = MEMBER_SUFFIX) -> Iterator[object]:
    """Yield each JSON line of the one archive member whose name ends with ``suffix``.

    The member is read as a stream; nothing is written to disk, so archive paths
    cannot escape anywhere. A line that is not valid JSON is yielded as its raw
    text and later excluded as malformed.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)]
        if len(members) != 1:
            raise ValueError(f"expected exactly one member ending in {suffix}, found {[m.name for m in members]}")
        fh = tar.extractfile(members[0])
        assert fh is not None
        for raw in fh:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield line


def longest_training_example(splits: data.Splits, count_tokens: Callable[[list[dict]], int], templates: prompts.Templates) -> tuple[int, str]:
    """(token count, example id) of the longest Train example as a full training chat."""
    best = (0, "")
    for ex in splits.train:
        n = count_tokens(prompts.training_messages(templates, ex))
        if n > best[0]:
            best = (n, ex.example_id)
    return best


def next_power_of_two_above(n: int) -> int:
    p = 1
    while p <= n:
        p *= 2
    return p


def chat_token_ids(tok, messages: list[dict]) -> list[int]:
    """Token ids of a chat; transformers 5 returns a mapping with ``input_ids``, older versions a list."""
    out = tok.apply_chat_template(messages, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def hf_token_counter(model: settings.ModelSettings) -> Callable[[list[dict]], int]:
    from fts.lora import load_tokenizer

    tok = load_tokenizer(model.name, model.revision)
    return lambda messages: len(chat_token_ids(tok, messages))


def build(
    tar_path: Path,
    url: str,
    accessed: str,
    stamp: repro.Stamp,
    count_tokens: Callable[[list[dict]], int],
    max_seq_len: int,
    root: Path = REPO_ROOT,
) -> tuple[int, dict]:
    """Run the pipeline and write every output; returns (exit status, counts payload)."""
    sha = sha256_of(tar_path)
    report = data.transform_all(iter_member_records(tar_path), SEED)
    for exc in report.exclusions:
        print(f"excluded {exc.example_id}: {exc.reason}")
    print(f"excluded total: {report.excluded_count}")
    splits = data.build_splits(report.examples, SEED)
    overlaps = data.check_overlap(splits)
    if overlaps:
        for pair, ids in overlaps.items():
            print(f"OVERLAP {pair}: {len(ids)} ids: {', '.join(ids)}", file=sys.stderr)
        print(f"FAIL: {data.overlap_total(overlaps)} examples appear in more than one split", file=sys.stderr)
        return 1, {}
    shared = data.shared_utterances(splits)
    for pair, n in shared.items():
        print(f"shared utterance texts {pair}: {n}")

    data.write_splits(splits, root / "data" / "massive_en")
    labels = data.labels_from_train(splits.train)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    longest, longest_id = longest_training_example(splits, count_tokens, prompts.load_templates(root / "config" / "prompts"))
    fits = longest <= max_seq_len
    payload = {
        "dataset": {"name": "MASSIVE", "version": "1.1", "locale": "en-US", "url": url, "sha256": sha, "accessed": accessed},
        "counts": data.split_counts(splits),
        "excluded": report.excluded_count,
        "exclusions": [{"id": e.example_id, "reason": e.reason} for e in report.exclusions],
        "overlap": {"pairs": overlaps, "total": 0},
        "shared_utterances": shared,
        "labels": {"intents": len(labels["intents"]), "slot_types": len(labels["slot_types"])},
        "longest_training_example": {
            "tokens": longest,
            "example_id": longest_id,
            "max_seq_len": max_seq_len,
            "fits": fits,
            "next_max_seq_len_if_not": None if fits else next_power_of_two_above(longest),
        },
    }
    repro.write_json(root / "results" / "data_counts.json", payload, stamp)
    envdoc.replace_section("Dataset", DATASET_SECTION.format(url=url, sha256=sha, accessed=accessed), root / "THIRD_PARTY.md")
    print(f"counts {payload['counts']}; longest training example {longest} tokens (max_seq_len {max_seq_len})")
    if not fits:
        print(f"NOTE: longest example exceeds max_seq_len; design.md rule gives {next_power_of_two_above(longest)}", file=sys.stderr)
    return 0, payload


def main(argv: list[str] | None = None) -> int:
    stamp = repro.start_run(seed=SEED)
    accessed = dt.date.today().isoformat()
    tar_path = download()
    model = settings.load_model_settings()
    train = settings.load_train_settings()
    status, _ = build(tar_path, URL, accessed, stamp, hf_token_counter(model), train.max_seq_len)
    return status
