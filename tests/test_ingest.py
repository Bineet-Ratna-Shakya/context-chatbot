from __future__ import annotations

from app import ingest as ingest_module
from app.ingest import load_documents, parse_frontmatter


def test_parse_frontmatter_extracts_keys():
    text = "---\ntitle: LangChain\nsource: https://x\n---\n\nBody line.\n"
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "LangChain"
    assert meta["source"] == "https://x"
    assert body == "Body line.\n"


def test_parse_frontmatter_without_block_returns_text():
    text = "# Heading\n\nBody.\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_load_documents_prefers_frontmatter_then_heading(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("---\ntitle: A\nsource: https://a\n---\n\nAlpha.\n")
    (tmp_path / "b.md").write_text("# Beta Doc\n\nBeta.\n")
    monkeypatch.setattr(ingest_module, "DOCS_DIR", tmp_path)

    docs = {d.metadata["title"]: d for d in load_documents()}

    assert docs["A"].metadata["source"] == "https://a"
    assert docs["A"].page_content == "Alpha.\n"
    assert "Beta Doc" in docs
    assert docs["Beta Doc"].metadata["source"].endswith("b.md")
