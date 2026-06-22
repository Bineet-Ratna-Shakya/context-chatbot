from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(ROOT / "docs")))
# Chroma persists to sqlite, which does not work on exFAT/network volumes. Override
# CHROMA_DIR to an internal-disk path when developing on such a drive; Docker is fine.
PERSIST_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    meta: dict[str, str] = {}
    for line in text[3:end].strip().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, text[end + 4:].lstrip("\n")


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))

        title = meta.get("title")
        if not title:
            for line in body.splitlines():
                if line.startswith("# "):
                    title = line.removeprefix("# ").strip()
                    break
        title = title or path.stem.replace("_", " ").title()
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            relative = str(path)
        source = meta.get("source") or relative

        documents.append(
            Document(page_content=body, metadata={"title": title, "source": source})
        )

    return documents


def main() -> None:
    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        print(f"Index already exists at {PERSIST_DIR}")
        return

    documents = load_documents()
    if not documents:
        raise SystemExit(
            f"No documents found in {DOCS_DIR}. Add *.md files or run "
            "uv run python -m app.snapshot to fetch Wikipedia pages."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=160)
    chunks = splitter.split_documents(documents)

    PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents into {PERSIST_DIR}")


if __name__ == "__main__":
    main()
