"""Unit tests for scripts/check_traceability.py (Req 10.4, 10.5)."""

from __future__ import annotations

from scripts import check_traceability as ct


def write(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_doc_numbers_skip_names_urls_and_labels():
    text = ("Test F1 0.8834 [0.8721, 0.8940] on 2,974 examples; P95 under 2000 ms at 95% of requests.\n"
            "Qwen2.5-1.5B-Instruct, gpt-6-luna, Q4_K_M, Ubuntu-24.04, scripts/verify_m5.sh, see https://example.com/v1.2/x 7.\n"
            "torch 2.14.1 costs $0.4318.")
    tokens = [t for _, t in ct.doc_numbers(text)]
    assert tokens == ["0.8834", "0.8721", "0.8940", "2974", "2000", "95", "7", "2.14.1", "0.4318"]
    assert ct.doc_numbers("a\nb 3")[0] == (2, "3")


def test_normalize():
    assert ct.normalize("2,974") == "2974" and ct.normalize("1,234,567") == "1234567" and ct.normalize("3.") == "3"
    assert ct.normalize("0,5") == "0,5"  # not a thousands separator


def test_check_flags_exactly_the_untraceable(tmp_path):
    src = write(tmp_path, "results/summary.md", "F1 0.8834 [0.8721, 0.8940]; 2974 examples; spend $0.4318")
    doc = write(tmp_path, "README.md", "F1 0.8834 on 2,974 examples cost $0.4318, but 0.9999 is made up.\nAlso 12.")
    ok, bad = ct.check_traceability(doc, [src])
    assert not ok and bad == ["line 1: 0.9999", "line 2: 12"]
    doc2 = write(tmp_path, "OK.md", "0.8721 and 0.8940")
    assert ct.check_traceability(doc2, [src]) == (True, [])


def test_default_sources_and_main(tmp_path, capsys):
    write(tmp_path, "results/summary.md", "0.5 and 7")
    write(tmp_path, "results/a.json", '{"x": 42}')
    write(tmp_path, "results/baselines/b.json", '{"y": 3}')
    write(tmp_path, "docs/ENVIRONMENT.md", "torch 2.14.1")
    sources = ct.default_sources(tmp_path)
    assert [p.name for p in sources] == ["summary.md", "ENVIRONMENT.md", "a.json", "b.json"]
    write(tmp_path, "README.md", "0.5, 42, 3, 2.14.1")
    write(tmp_path, "MODEL_CARD.md", "7 and 8")
    assert ct.main(["README.md"], root=tmp_path) == 0
    assert ct.main([], root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "UNTRACEABLE MODEL_CARD.md line 1: 8" in out
    assert ct.main(["missing.md"], root=tmp_path) == 1
