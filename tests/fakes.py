from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        return AIMessage(content=self.reply)


class FakeVectorStore:
    def __init__(self, scored: list[tuple[Document, float]]) -> None:
        self.scored = scored
        self.queries: list[str] = []

    def similarity_search_with_score(self, query: str, k: int = 4):
        self.queries.append(query)
        return self.scored[:k]


class FakeGraph:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def ainvoke(self, payload, config=None):
        if self.fail:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return {
            "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
            "sources": [{"title": "LangGraph", "source": "https://x", "snippet": "s"}],
        }
