from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.graph import build_graph
from app.ingest import COLLECTION_NAME, PERSIST_DIR, get_embeddings

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()
]

_request_times: dict[str, deque[float]] = defaultdict(deque)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=128)


class Source(BaseModel):
    title: str
    source: str
    snippet: str
    page: int | None = None


class ChatResponse(BaseModel):
    response: str
    sources: list[Source]


def rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    times = _request_times[client]
    while times and now - times[0] > 60:
        times.popleft()
    if len(times) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    times.append(now)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def token_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    if not PERSIST_DIR.exists():
        raise RuntimeError("Index not found — run uv run python -m app.ingest.")

    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    app.state.graph = build_graph(vectorstore)
    yield


app = FastAPI(title="Context Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        result = await request.app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=payload.message)]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ai_messages = [message for message in result["messages"] if isinstance(message, AIMessage)]
    return ChatResponse(
        response=ai_messages[-1].content if ai_messages else "",
        sources=result.get("sources", []),
    )


@app.post("/chat/stream", dependencies=[Depends(rate_limit)])
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    async def events():
        sources_sent = False
        try:
            async for mode, chunk in request.app.state.graph.astream(
                {"messages": [HumanMessage(content=payload.message)]},
                config={"configurable": {"thread_id": payload.thread_id}},
                stream_mode=["updates", "messages"],
            ):
                if mode == "updates" and "retrieve" in chunk:
                    yield sse("sources", chunk["retrieve"].get("sources", []))
                    sources_sent = True
                elif mode == "messages":
                    message, metadata = chunk
                    if metadata.get("langgraph_node") == "generate":
                        token = token_text(getattr(message, "content", ""))
                        if token:
                            yield sse("token", token)

            if not sources_sent:
                yield sse("sources", [])
            yield sse("done", True)
        except Exception as exc:
            yield sse("error", str(exc))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
