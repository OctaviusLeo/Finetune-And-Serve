"""Unit tests for the baselines runner with inference and HTTP faked (task 2.12; Req 3.4, 3.6)."""

from __future__ import annotations

import json
import shutil

import httpx
import pytest

from fts import baselines, data, exemplars, jev_extract, prompts, repro, spend
from fts.infer import Prediction
from fts.jev_client import JevClient

STAMP = repro.Stamp(command="python -m fts.baselines", seed=0, commit="3" * 40)
GOLD = [
    '{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]}',
    '{"intent": "play_music", "slots": []}',
    '{"intent": "weather_query", "slots": [{"type": "place_name", "value": "paris"}]}',
]
UTTS = ["wake me at five am", "play something", "weather in paris"]
TEST = tuple(data.Example(str(i), "test", u, g) for i, (u, g) in enumerate(zip(UTTS, GOLD)))
TRAIN = (data.Example("100", "train", "wake me", GOLD[0]),)
SPLITS = data.Splits(TRAIN, (), TEST)
LABELS = {"intents": ["alarm_set", "play_music", "weather_query"], "slot_types": ["place_name", "time"]}
T = prompts.load_templates()
PLAN = baselines.JevPlan(labels=LABELS, exemplars=TRAIN, templates=jev_extract.load_templates(), max_span_tokens=6, per_request=100)
GOLD_BY_UTT = {e.inputs: json.loads(e.target) for e in TEST}


def perfect_engine(examples, build, batch_size):
    gold = {e.example_id: e.target for e in TEST}
    return [Prediction(e.example_id, gold[e.example_id], 50, 10) for e in examples]


def jev_oracle(request, input_tokens=None):
    """A fake Jev that answers from the gold labels: the gold intent, the gold type for a gold span, none otherwise."""
    body = json.loads(request.content)
    gold = GOLD_BY_UTT[body["state"]["request"]]
    answers = {}
    for qid, q in body["questions"].items():
        if qid == "intent":
            pick = gold["intent"]
        else:
            span = q["instructions"].split('"')[1]
            pick = next((sl["type"] for sl in gold["slots"] if sl["value"] == span), "none")
        answers[qid] = {"type": "choice", "choice": pick, "probabilities": {o: (1.0 if o == pick else 0.0) for o in q["criteria"]}, "confidence": 1.0}
    tokens = len(request.content) // 3 if input_tokens is None else input_tokens
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 50}})


def hosted_client(ledger, handler, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    return JevClient("jev-1.13.0", transport=httpx.MockTransport(handler), ledger=ledger, ledger_tag="hosted_few_shot")


def make_ledger(tmp_path, cap=2.0):
    return spend.SpendLedger(tmp_path / "ledger.jsonl", cap_usd=cap, prices=spend.Prices(0.042, 0.0))


def test_run_baselines_records_templates_spend_and_shared_test_ids(tmp_path, monkeypatch):
    ledger = make_ledger(tmp_path)
    client = hosted_client(ledger, lambda r: jev_oracle(r, input_tokens=1000), monkeypatch)
    results = baselines.run_baselines(
        SPLITS, TRAIN, 2.0, 0, engine=perfect_engine, client=client, ledger=ledger, labels=LABELS, templates=T, plan=PLAN,
        cache_path=tmp_path / "pred" / "hosted.cache.jsonl", predictions_dir=tmp_path / "pred",
    )
    assert [r.name for r in results] == list(baselines.NAMES)
    assert all(r.metric.point == 1.0 and r.not_scored == 0 and r.scored == 3 for r in results)
    assert len({r.test_ids_sha256 for r in results}) == 1 and results[0].test_ids_sha256 == baselines.split_digest(TEST)
    assert results[0].prompt_template == "config/prompts/system.txt, config/prompts/zero_shot.txt"
    assert "config/prompts/few_shot.txt" in results[1].prompt_template
    assert results[2].prompt_template == "config/prompts/jev.yaml, config/labels.json, config/few_shot_exemplars.json"
    assert results[0].spend_usd == 0.0 and results[2].spend_usd == pytest.approx(3 * 1000 * 0.042 / 1e6)
    d = results[2].details
    assert d["response_models"] == ["jev-1.13.0"] and d["requests"] == 3 and d["input_tokens"] == 3000 and d["max_span_tokens"] == 6
    spans = sum(len(jev_extract.candidate_spans(u, 6)) for u in UTTS)
    assert d["questions"] == spans + 3
    assert (tmp_path / "pred" / "base_zero_shot.jsonl").exists() and (tmp_path / "pred" / "hosted_few_shot.jsonl").exists()


def test_run_baselines_rejects_mismatched_cap(tmp_path):
    ledger = make_ledger(tmp_path, cap=1.0)
    with pytest.raises(ValueError):
        baselines.run_baselines(SPLITS, TRAIN, 2.0, 0, engine=perfect_engine, client=None, ledger=ledger, labels=LABELS,
                                templates=T, plan=PLAN, cache_path=tmp_path / "c.jsonl")


def test_spend_cap_stops_the_hosted_baseline(tmp_path, monkeypatch):
    # the cap fits the first request's worst case and nothing after it
    probe = hosted_client(None, jev_oracle, monkeypatch)
    spans, batches = jev_extract.question_batches(TEST[0].inputs, LABELS, PLAN.templates, 6, 100)
    worst = spend.estimate_text_tokens(probe.body(jev_extract.build_state(TEST[0].inputs, TRAIN), batches[0]))
    ledger = make_ledger(tmp_path, cap=spend.Prices(0.042, 0.0).cost(worst, 0) * 1.01)
    calls = []
    client = hosted_client(ledger, lambda r: calls.append(r) or jev_oracle(r, input_tokens=worst), monkeypatch)
    cache = tmp_path / "hosted.cache.jsonl"
    with pytest.raises(spend.SpendCapExceeded) as excinfo:
        baselines.run_hosted(TEST, client, ledger, PLAN, 0, cache)
    assert len(calls) == 1 and ledger.requests == 1
    assert excinfo.value.spent_usd == pytest.approx(ledger.total_usd) and ledger.total_usd <= ledger.cap_usd
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 1


def test_hosted_resumes_from_cache_without_resending(tmp_path, monkeypatch):
    cache = tmp_path / "hosted.cache.jsonl"
    old = {"example_id": "0", "text": GOLD[0], "input_tokens": 5, "output_tokens": 1, "requests": 1, "questions": 2, "models": ["jev-1.12.0"], "answers": {}}
    cache.write_text(json.dumps(old) + "\n", encoding="utf-8")
    ledger = make_ledger(tmp_path)
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["state"]["request"])
        return jev_oracle(request)

    logs = []
    result = baselines.run_hosted(TEST, hosted_client(ledger, handler, monkeypatch), ledger, PLAN, 0, cache, log=logs.append)
    assert sent == UTTS[1:] and result.metric.point == 1.0
    assert result.details["response_models"] == ["jev-1.12.0", "jev-1.13.0"]
    rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["answers"]["intent"] == ["weather_query", 1.0] and rows[-1]["text"] == GOLD[2]


def test_long_requests_are_split_across_requests(tmp_path, monkeypatch):
    ledger = make_ledger(tmp_path)
    plan = baselines.JevPlan(labels=LABELS, exemplars=TRAIN, templates=PLAN.templates, max_span_tokens=6, per_request=4)
    sizes = []

    def handler(request):
        sizes.append(len(json.loads(request.content)["questions"]))
        return jev_oracle(request)

    row = baselines.jev_predict(hosted_client(ledger, handler, monkeypatch), TEST[0], plan)
    n_spans = len(jev_extract.candidate_spans(TEST[0].inputs, 6))
    assert row["requests"] == len(sizes) == -(-n_spans // 4) and sum(sizes) == n_spans + 1 and row["text"] == GOLD[0]


def test_not_scored_examples_are_counted():
    bad_test = TEST + (data.Example("9", "test", "x", "not json"),)
    result = baselines.evaluate("base_zero_shot", bad_test, perfect_engine(TEST, None, 8), seed=0)
    assert result.not_scored == 1 and result.not_scored_examples[0]["id"] == "9" and result.n_test == 4
    with pytest.raises(ValueError):
        baselines.evaluate("base_zero_shot", (bad_test[-1],), [], seed=0)


def test_combine_requires_all_three_and_one_test_split(tmp_path):
    ledger_summary = {"spend_usd": 0.5, "spend_cap_usd": 2.0, "prompt_price_per_mtok": 0.042, "completion_price_per_mtok": 0.0}
    for name in baselines.NAMES[:2]:
        repro.write_json(tmp_path / f"{name}.json", {"name": name, "test_ids_sha256": "abc", "n_test": 3, "spend_usd": 0.0, "details": {}}, STAMP)
    assert baselines.combine(tmp_path) is None
    repro.write_json(tmp_path / "hosted_few_shot.json", {"name": "hosted_few_shot", "test_ids_sha256": "abc", "n_test": 3, "spend_usd": 0.4,
                                                         "details": {"ledger": ledger_summary}}, STAMP)
    combined = baselines.combine(tmp_path)
    assert [b["name"] for b in combined["baselines"]] == list(baselines.NAMES)
    assert combined["hosted_spend"]["within_cap"] is True and combined["n_test"] == 3
    repro.write_json(tmp_path / "base_few_shot.json", {"name": "base_few_shot", "test_ids_sha256": "zzz", "n_test": 3, "spend_usd": 0.0, "details": {}}, STAMP)
    with pytest.raises(ValueError, match="different Test_Splits"):
        baselines.combine(tmp_path)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    shutil.copytree(baselines.REPO_ROOT / "config" / "prompts", r / "config" / "prompts")
    for f in ("hosted.yaml", "labels.json", "model.yaml"):
        shutil.copy(baselines.REPO_ROOT / "config" / f, r / "config" / f)
    data.write_splits(SPLITS, r / "data" / "massive_en")
    exemplars.save_exemplars(TRAIN, r / "config" / "few_shot_exemplars.json")
    monkeypatch.setattr(baselines.repro, "start_run", lambda seed: STAMP)
    return r


def test_main_local_then_hosted_then_combined(root, monkeypatch):
    assert baselines.main(["--baseline", "local"], root=root, engine=perfect_engine) == 0
    assert (root / "results" / "baselines" / "base_few_shot.json").exists() and not (root / "results" / "baselines.json").exists()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert baselines.main(["--baseline", "hosted_few_shot"], root=root) == 2
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    assert baselines.main(["--baseline", "hosted_few_shot"], root=root, transport=httpx.MockTransport(jev_oracle)) == 0
    combined = json.loads((root / "results" / "baselines.json").read_text(encoding="utf-8"))
    assert combined["stamp"]["commit"] == STAMP.commit and len(combined["baselines"]) == 3
    assert combined["hosted_spend"]["spend_cap_usd"] == 2.0


def test_main_hosted_failures_exit_1(root, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    assert baselines.main(["--baseline", "hosted_few_shot"], root=root, transport=httpx.MockTransport(lambda r: httpx.Response(401))) == 1
    assert "hosted request failed" in capsys.readouterr().err
    cfg = root / "config" / "hosted.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("spend_cap_usd: 2.00", "spend_cap_usd: 0.0"), encoding="utf-8")
    assert baselines.main(["--baseline", "hosted_few_shot"], root=root, transport=httpx.MockTransport(jev_oracle)) == 1
    assert "Spend reached $0.000000; cap $0.00" in capsys.readouterr().err


def test_default_engine_loads_once(monkeypatch):
    loads = []
    monkeypatch.setattr(baselines.infer, "load_for_inference", lambda name, rev, device: loads.append(name) or ("model", "tok"))
    monkeypatch.setattr(baselines.infer, "predict", lambda m, t, exs, build, bs, progress: [progress(len(exs), len(exs))] and [])
    engine = baselines.default_engine(baselines.settings.load_model_settings(), device="cpu")
    engine(TEST, None, 8)
    engine(TEST, None, 8)
    assert loads == ["Qwen/Qwen2.5-1.5B-Instruct"]
