---
title: Context Chatbot Architecture
source: docs/project_architecture.md
---

# Context Chatbot Architecture

This document describes how this project works, so the chatbot can answer questions
about its own design.

## Overview

The project is a context-aware retrieval-augmented generation (RAG) chatbot built
with LangGraph for control flow, FastAPI for the API, ChromaDB for vector search, and
a small vanilla web UI. It answers questions about LangChain, LangGraph, and
retrieval-augmented generation using a local document corpus.

## Retrieval pipeline

Documents are ingested from local Markdown files. Wikipedia pages (LangChain and
retrieval-augmented generation) are snapshotted into the corpus ahead of time so the
build is reproducible and does not depend on the live Wikipedia API. Text is split
into overlapping chunks of about 1200 characters with 160 characters of overlap,
embedded with the `sentence-transformers/all-MiniLM-L6-v2` model, and stored in a
Chroma vector store, a persistent sqlite-backed database. The collection uses cosine
distance.

At query time the user's question is embedded with the same model and compared
against the index. Chunks beyond a configurable cosine distance threshold are dropped,
so an off-topic question returns no context and the model is instructed to say it
does not know rather than hallucinate.

## Graph flow

The LangGraph state machine has three nodes connected in a line: contextualize,
retrieve, and generate. The contextualize node rewrites follow-up questions into a
standalone search query using earlier turns, retrieve runs the Chroma similarity
search and formats sources, and generate produces the final answer from the
retrieved context plus the full conversation history. Conversation state is kept per
thread with a LangGraph checkpointer, so follow-up questions resolve references like
"it" to the right subject.

## API and UI

FastAPI exposes a health check, a JSON chat endpoint, and a streaming chat endpoint
that sends answer tokens and sources over Server-Sent Events. The web UI is served
from the same origin and consumes the streaming endpoint; it never talks to the LLM
directly. The Groq API provides the chat models, with a small fast model for query
rewriting and a larger model for answers.

## Deployment

The application is containerized with a multi-stage Docker build that uses uv for
dependency management, builds the Chroma index from the committed corpus at build
time, and caches the embedding model. Only the Groq API key is required at run time;
no secrets are baked into the image.
