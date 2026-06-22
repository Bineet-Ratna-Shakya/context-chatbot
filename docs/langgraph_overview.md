---
title: LangGraph Overview
source: https://langchain-ai.github.io/langgraph/
---

# LangGraph Overview

LangGraph is an open-source framework developed by LangChain Inc. for building
stateful, multi-step applications with large language models. It extends the
LangChain ecosystem with graph-based control flow, persistent state, memory,
human-in-the-loop review, and cyclic execution patterns. There is no standalone
Wikipedia article for LangGraph; this document is the authored reference used by
this project.

## Why LangGraph instead of a plain chain

A LangChain chain is a fixed, linear sequence of steps. That is enough for simple
prompt pipelines, but it breaks down when an application needs branching, loops,
retries, or memory that persists across turns. LangGraph models the application as
a state machine: you declare a typed state object, write nodes that read and update
that state, and connect them with edges. The framework runs the graph, threads the
state through each node, and persists checkpoints so a conversation can resume
exactly where it left off.

## Core concepts

- **State**: a typed dictionary (often a `TypedDict`) that flows through the graph.
  Fields can declare reducers, such as `add_messages`, which appends new messages
  to the conversation history instead of overwriting it.
- **Nodes**: plain functions that receive the current state and return a partial
  update. A node might rewrite a question, retrieve documents, or call an LLM.
- **Edges**: connections between nodes. Edges can be unconditional or conditional,
  and they may form cycles for agent-style loops.
- **StateGraph**: the builder used to register nodes and edges and then compile the
  runnable graph.
- **Checkpointers**: storage backends that persist state per thread. `MemorySaver`
  keeps state in process memory, while durable savers (SQLite, Postgres) survive
  restarts and scale across workers.

## Common use cases

LangGraph is used for multi-step assistants, agent workflows, retrieval-augmented
generation pipelines, tool-using systems, and any application that must preserve
state across turns or make control-flow decisions between LLM calls.

## Conversational RAG with LangGraph

In a conversational RAG application, LangGraph separates the work into clear nodes:
rewriting the user's current message into a standalone query using prior turns,
retrieving evidence from a vector store, and generating an answer from the retrieved
context plus the original conversation history. Keeping these steps as distinct
nodes makes the flow easy to read, test, and extend, and lets memory and retrieval
be reasoned about independently.

Sources:

- https://langchain-ai.github.io/langgraph/
- https://github.com/langchain-ai/langgraph
- https://www.langchain.com/langgraph
