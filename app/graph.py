from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: str
    sources: list[dict[str, Any]]
    ambiguous: bool


ANSWER_MODEL = os.getenv("ANSWER_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
# Chroma cosine distance (1 - cosine similarity), range [0, 2]. ~0.7 keeps on-topic
# chunks and drops off-topic ones so the model abstains instead of hallucinating.
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.7"))

PRONOUNS = {
    "it", "this", "that", "they", "them", "he", "she", "him", "her",
    "its", "their", "theirs", "his", "hers",
}

_answer_llm: ChatGroq | None = None
_fast_llm: ChatGroq | None = None


def get_llm(model: str) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")
    return ChatGroq(model=model, temperature=0)


def answer_llm() -> ChatGroq:
    global _answer_llm
    if _answer_llm is None:
        _answer_llm = get_llm(ANSWER_MODEL)
    return _answer_llm


def fast_llm() -> ChatGroq:
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = get_llm(FAST_MODEL)
    return _fast_llm


def latest_human(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def source_from_doc(doc: Any) -> dict[str, Any]:
    metadata = doc.metadata
    snippet = " ".join(doc.page_content.split())[:260]
    source = {
        "title": metadata.get("title") or "Untitled",
        "source": metadata.get("source") or "",
        "snippet": snippet,
    }
    page = metadata.get("page")
    if page is not None:
        source["page"] = page
    return source


def retrieve_context(vectorstore: VectorStore, question: str) -> tuple[str, list[dict[str, Any]]]:
    results = vectorstore.similarity_search_with_score(question, k=RETRIEVAL_K)
    docs = [doc for doc, score in results if score <= RETRIEVAL_MAX_DISTANCE]
    context = "\n\n".join(
        f"Source {index}: {doc.metadata.get('title', 'Untitled')}\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )
    return context, [source_from_doc(doc) for doc in docs]


def build_graph(vectorstore: VectorStore):
    def contextualize(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        question = latest_human(messages)
        prior_questions = [
            str(message.content)
            for message in messages[:-1]
            if isinstance(message, HumanMessage)
        ]

        if not prior_questions:
            words = question.lower().replace("?", " ").replace(",", " ").split()
            ambiguous = any(word in PRONOUNS for word in words)
            return {"question": question, "ambiguous": ambiguous}

        prompt = [
            SystemMessage(
                content=(
                    "Rewrite the user's latest message as a standalone search query. "
                    "Use earlier user messages to resolve references. Return only the query."
                )
            ),
            HumanMessage(
                content=(
                    "Earlier user messages:\n"
                    + "\n".join(f"- {item}" for item in prior_questions[-6:])
                    + f"\n\nLatest user message:\n{question}"
                )
            ),
        ]
        rewritten = fast_llm().invoke(prompt, config=config)
        return {"question": str(rewritten.content).strip() or question, "ambiguous": False}

    def retrieve(state: ChatState) -> dict[str, Any]:
        if state.get("ambiguous"):
            return {"context": "", "sources": []}

        context, sources = retrieve_context(vectorstore, state["question"])
        return {"context": context, "sources": sources}

    def generate(state: ChatState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        if state.get("ambiguous"):
            response = answer_llm().invoke(
                [
                    SystemMessage(
                        content=(
                            "The user's message has an unresolved reference and there is "
                            "no prior conversation. Ask one short clarification question."
                        )
                    ),
                    *state.get("messages", []),
                ],
                config=config,
            )
            return {"messages": [response]}

        system_prompt = (
            "You are a concise RAG assistant. Answer using the context below as the source "
            "of truth. If the indexed context does not answer the question, say you do not "
            "know from the provided documents. Do not resolve ambiguous pronouns from "
            "retrieved context alone; rely on the conversation history for the referent.\n\n"
            f"Context:\n{state.get('context', '')}"
        )
        response = answer_llm().invoke(
            [SystemMessage(content=system_prompt), *state.get("messages", [])],
            config=config,
        )
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=MemorySaver())
