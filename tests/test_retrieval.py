from __future__ import annotations

from langchain_core.documents import Document

from app import graph as graph_module
from app.graph import retrieve_context, source_from_doc

from .fakes import FakeVectorStore


def test_retrieve_context_filters_far_docs(monkeypatch):
    monkeypatch.setattr(graph_module, "RETRIEVAL_MAX_DISTANCE", 1.0)
    near = Document(
        page_content="LangGraph builds stateful graphs.",
        metadata={"title": "LangGraph", "source": "url"},
    )
    far = Document(
        page_content="A recipe for sourdough bread.",
        metadata={"title": "Bread", "source": "url2"},
    )
    store = FakeVectorStore([(near, 0.5), (far, 1.5)])

    context, sources = retrieve_context(store, "what is langgraph")

    assert "LangGraph builds" in context
    assert "sourdough" not in context
    assert len(sources) == 1
    assert sources[0]["title"] == "LangGraph"


def test_retrieve_context_empty_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(graph_module, "RETRIEVAL_MAX_DISTANCE", 1.0)
    far = Document(page_content="Unrelated text.", metadata={"title": "X", "source": "y"})
    store = FakeVectorStore([(far, 1.9)])

    context, sources = retrieve_context(store, "off topic question")

    assert context == ""
    assert sources == []


def test_source_from_doc_omits_null_page():
    doc = Document(page_content="a b c", metadata={"title": "T", "source": "s"})
    assert "page" not in source_from_doc(doc)


def test_source_from_doc_keeps_page_when_present():
    doc = Document(page_content="a b c", metadata={"title": "T", "source": "s", "page": 3})
    assert source_from_doc(doc)["page"] == 3
