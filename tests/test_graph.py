from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app import graph as graph_module
from app.graph import build_graph

from .fakes import FakeLLM, FakeVectorStore


def _patch_llms(monkeypatch, fast: FakeLLM, answer: FakeLLM) -> None:
    monkeypatch.setattr(graph_module, "fast_llm", lambda: fast)
    monkeypatch.setattr(graph_module, "answer_llm", lambda: answer)


def test_first_turn_pronoun_is_ambiguous_and_skips_retrieval(monkeypatch):
    fast = FakeLLM("rewritten")
    answer = FakeLLM("Which tool do you mean?")
    _patch_llms(monkeypatch, fast, answer)
    store = FakeVectorStore([(Document(page_content="c", metadata={"title": "T", "source": "s"}), 0.3)])

    graph = build_graph(store)
    out = graph.invoke(
        {"messages": [HumanMessage(content="What is it?")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert out["sources"] == []
    assert fast.calls == []
    assert store.queries == []


def test_first_turn_plain_question_retrieves_without_rewrite(monkeypatch):
    fast = FakeLLM("rewritten")
    answer = FakeLLM("LangGraph is a framework.")
    _patch_llms(monkeypatch, fast, answer)
    doc = Document(page_content="LangGraph builds graphs.", metadata={"title": "LangGraph", "source": "url"})
    store = FakeVectorStore([(doc, 0.4)])

    graph = build_graph(store)
    graph.invoke(
        {"messages": [HumanMessage(content="What is LangGraph?")]},
        config={"configurable": {"thread_id": "t2"}},
    )

    assert fast.calls == []
    assert store.queries == ["What is LangGraph?"]


def test_followup_rewrites_query_and_is_not_ambiguous(monkeypatch):
    fast = FakeLLM("LangGraph developer")
    answer = FakeLLM("It is developed by LangChain.")
    _patch_llms(monkeypatch, fast, answer)
    doc = Document(
        page_content="LangGraph is developed by LangChain.",
        metadata={"title": "LangGraph", "source": "url"},
    )
    store = FakeVectorStore([(doc, 0.4)])

    graph = build_graph(store)
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke({"messages": [HumanMessage(content="What is LangGraph?")]}, config=config)
    out = graph.invoke({"messages": [HumanMessage(content="Who develops it?")]}, config=config)

    assert fast.calls
    assert store.queries[-1] == "LangGraph developer"
    assert out["sources"][0]["title"] == "LangGraph"
