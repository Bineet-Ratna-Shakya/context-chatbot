from __future__ import annotations

import pytest

from app.ingest import COLLECTION_NAME, PERSIST_DIR, get_embeddings
from evals.dataset import DATASET

# This is an integration check against the real Chroma index, so it only runs
# where the index has been built (locally after `app.ingest`); it is skipped in
# CI, where the unit tests with fakes provide coverage instead.
pytestmark = pytest.mark.skipif(
    not PERSIST_DIR.exists(), reason="Chroma index not built (run app.ingest)"
)


@pytest.fixture(scope="module")
def vectorstore():
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


@pytest.mark.parametrize("item", DATASET, ids=lambda i: i["question"])
def test_retrieval_matches_expected_source(item, vectorstore):
    from app.graph import retrieve_context

    _, sources = retrieve_context(vectorstore, item["question"])
    titles = [source["title"] for source in sources]

    if item["expected_source"] is None:
        assert sources == [], f"off-topic question leaked sources: {titles}"
    else:
        assert item["expected_source"] in titles, (
            f"{item['question']!r} expected {item['expected_source']!r}, got {titles}"
        )
