from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import answer_llm, build_graph, retrieve_context
from app.ingest import COLLECTION_NAME, PERSIST_DIR, get_embeddings
from evals.dataset import DATASET


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def eval_retrieval(vectorstore: Chroma) -> None:
    hits = on_topic = abstained = off_topic = 0
    for item in DATASET:
        _, sources = retrieve_context(vectorstore, item["question"])
        titles = [source["title"] for source in sources]
        if item["expected_source"] is None:
            off_topic += 1
            ok = len(sources) == 0
            abstained += int(ok)
            print(f"  [off-topic] {item['question']!r} -> {len(sources)} sources ({'ok' if ok else 'LEAK'})")
        else:
            on_topic += 1
            ok = item["expected_source"] in titles
            hits += int(ok)
            print(f"  {'HIT ' if ok else 'MISS'} {item['question']!r} -> {titles}")

    print(f"\n  retrieval hit rate:    {hits}/{on_topic}")
    print(f"  off-topic abstention:  {abstained}/{off_topic}")


def judge_faithful(question: str, context: str, answer: str) -> bool:
    prompt = [
        SystemMessage(
            content=(
                "You are a strict grader for a RAG system. Given a question, the "
                "retrieved context, and an answer, reply with only YES or NO. Reply YES "
                "if the answer is fully supported by the context, or if the context is "
                "empty and the answer correctly says it does not know. Otherwise NO."
            )
        ),
        HumanMessage(
            content=f"Question: {question}\n\nContext:\n{context or '(empty)'}\n\nAnswer:\n{answer}"
        ),
    ]
    return str(answer_llm().invoke(prompt).content).strip().upper().startswith("YES")


def eval_generation(vectorstore: Chroma) -> None:
    graph = build_graph(vectorstore)
    covered = total_kw = faithful = 0
    for index, item in enumerate(DATASET):
        result = graph.invoke(
            {"messages": [HumanMessage(content=item["question"])]},
            config={"configurable": {"thread_id": f"eval-{index}"}},
        )
        ai = [message for message in result["messages"] if isinstance(message, AIMessage)]
        answer = ai[-1].content if ai else ""
        context = result.get("context", "")

        answer_lower = answer.lower()
        kw_hits = sum(1 for kw in item["expected_keywords"] if kw.lower() in answer_lower)
        covered += kw_hits
        total_kw += len(item["expected_keywords"])
        faithful += int(judge_faithful(item["question"], context, answer))

        print(f"  Q: {item['question']!r}")
        print(f"    A: {answer[:110].strip()}...")
        print(f"    keywords {kw_hits}/{len(item['expected_keywords'])}")

    print(f"\n  answer keyword coverage:  {covered}/{total_kw}")
    print(f"  faithfulness (LLM-judge): {faithful}/{len(DATASET)}")


def main() -> None:
    load_dotenv()
    vectorstore = load_vectorstore()

    print("== Retrieval eval ==")
    eval_retrieval(vectorstore)

    if not os.getenv("GROQ_API_KEY"):
        print("\n(Set GROQ_API_KEY to also run the generation eval.)")
        return

    print("\n== Generation eval ==")
    eval_generation(vectorstore)


if __name__ == "__main__":
    main()
