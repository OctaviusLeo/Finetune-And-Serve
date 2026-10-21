"""Unit tests for README/Model_Card generation (fts.docs) and its traceability."""

from __future__ import annotations

import json
import shutil

import pytest

from fts import docs, report, repro
from scripts.check_traceability import check_traceability, default_sources
from tests.results_fixture import STAMP, write_results


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    shutil.copytree(docs.REPO_ROOT / "config", r / "config")
    (r / "docs").mkdir()
    for t in ("README.template.md", "MODEL_CARD.template.md"):
        shutil.copy(docs.REPO_ROOT / "docs" / t, r / "docs" / t)
    shutil.copy(docs.REPO_ROOT / "docs" / "ENVIRONMENT.md", r / "docs" / "ENVIRONMENT.md")
    write_results(r / "results")
    report.write_results_table(r / "results", r / "results" / "summary.md", repro.to_dict(STAMP))  # build_summary runs before build_docs
    monkeypatch.setattr(repro, "start_run", lambda seed: STAMP)
    return r


def test_render_fills_placeholders_and_rejects_unknown():
    assert docs.render("F1 {{a}} and {{b}}", {"a": "0.9", "b": "x"}) == "F1 0.9 and x"
    with pytest.raises(KeyError, match="'missing'"):
        docs.render("{{missing}}", {})


def test_doc_facts_come_from_configs(root):
    f = docs.doc_facts(root)
    assert f["max_seq_len"] == 256 and f["effective_batch"] == 16 and f["spend_cap_usd"] == "2.00"
    assert f["final_seeds"] == "1, 2, 3" and f["loadtest_levels"] == "1, 2, 4, 8, 16, 32" and f["serve_port"] == 8080
    assert f["n_exemplars"] == 10 and f["dataset_license"] == "CC BY 4.0" and f["base_model_license"] == "Apache-2.0"


def test_main_generates_both_documents_and_every_number_is_traceable(root):
    assert docs.main(root=root) == 0
    readme = (root / "README.md").read_text(encoding="utf-8")
    card = (root / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "{{" not in readme and "{{" not in card
    assert "[MIT License](LICENSE)" in readme and "not published" in readme and "not published" in card
    assert "0.8812 ± 0.0022" in readme and "0.7012 [0.6912, 0.7112]" in readme and "results/plots/loss_curves.png" in readme
    assert "`bash scripts/test_unit.sh`" in readme and "`python -m fts.smoke_eval`" in readme and "`bash scripts/serve.sh`" in readme
    assert "wrong_intent" in card and "151" in card and "CC BY 4.0" in card and "Apache-2.0" in card
    facts = json.loads((root / "results" / "doc_facts.json").read_text(encoding="utf-8"))
    assert facts["stamp"]["commit"] == STAMP.commit
    sources = default_sources(root)
    for doc in ("README.md", "MODEL_CARD.md"):
        ok, bad = check_traceability(root / doc, sources)
        assert ok, bad


def test_example_fallback_when_no_example_saved(root):
    ea = json.loads((root / "results" / "error_analysis.json").read_text(encoding="utf-8"))
    ea["examples"] = {}
    (root / "results" / "error_analysis.json").write_text(json.dumps(ea), encoding="utf-8")
    assert docs.facts(root)["ea_example"] == "no example was saved"
