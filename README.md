# Context-Aware RAG Chatbot

A small retrieval-augmented chatbot built with FastAPI and LangGraph. It answers
questions about LangChain, LangGraph and RAG from a local set of documents, keeps
track of the conversation so follow-up questions work, and streams the answer to a
simple web page. Search is handled by ChromaDB and the answers come from Groq.

- Docker image: https://hub.docker.com/r/soulpiee/context-chatbot
- GitHub repo: https://github.com/Bineet-Ratna-Shakya/context-chatbot
- Demo video: TODO

## Run

```bash
docker run --rm -p 8000:8000 -e GROQ_API_KEY=your_key_here soulpiee/context-chatbot:latest
```

Open http://localhost:8000. You'll need a free Groq key, which you can get at
https://console.groq.com/keys.

To run it from source instead:

```bash
uv sync
uv run python -m app.ingest
GROQ_API_KEY=your_key uv run uvicorn app.main:app
```

The ingest step builds the vector index from the files in docs/, so run it once
before starting the server.

## How it works

The chatbot is a LangGraph graph with three steps. It first rewrites the latest
message into a standalone search query using the earlier turns, so a vague follow-up
like "who develops it?" turns into something that can actually be searched. It then
runs a similarity search in ChromaDB and keeps only the chunks that are close enough,
which means an off-topic question simply finds nothing. Whatever it finds, together
with the conversation so far, goes to Groq, and the model writes the answer from that.

Conversation history is saved per session with a LangGraph checkpointer, and that is
what makes the follow-up questions work. The text is embedded with the
all-MiniLM-L6-v2 model before it goes into ChromaDB.

## What's in the index

The document set is four Markdown files in docs/, copied from a few different sources:
the LangChain and Retrieval-augmented generation pages from Wikipedia, a LangGraph
overview, and a note on the project's architecture.

At build time these are split into about 25 chunks and stored in ChromaDB with each
chunk's title and source URL, so every answer can cite where it came from.

## API

The two chat endpoints are POST /chat, which returns the full answer and its sources
as JSON, and POST /chat/stream, which streams the answer over SSE and is what the web
UI talks to. There is also a health check at GET /health, the UI itself at GET /, and
auto-generated Swagger docs at /docs.

## Config

The only setting you have to provide is GROQ_API_KEY. You can get one for free at
https://console.groq.com/keys. The rest, like which models to use and how many chunks
to pull back, have defaults and can be changed through environment variables if you
want. The full list is in .env.example.

## Tests

```bash
uv run pytest
uv run python -m evals.run_eval
```

The first runs the unit tests and a retrieval-quality check. The second runs a small
eval that scores retrieval and answer quality against a labeled set of questions.
