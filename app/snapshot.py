from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

from langchain_community.document_loaders import WikipediaLoader

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(ROOT / "docs")))
# Only pages that resolve to a real standalone article. "LangGraph" has no
# Wikipedia page (the query returns the LangChain article), so it is covered by
# an authored doc instead.
WIKIPEDIA_PAGES = tuple(
    page.strip()
    for page in os.getenv(
        "WIKIPEDIA_PAGES", "LangChain,Retrieval-augmented generation"
    ).split(",")
    if page.strip()
)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for title in WIKIPEDIA_PAGES:
        loaded = []
        for attempt in range(3):
            try:
                loaded = WikipediaLoader(
                    query=title, load_max_docs=1, doc_content_chars_max=20000
                ).load()
                break
            except Exception as exc:
                print(f"Attempt {attempt + 1} for {title} failed: {exc}")
        if not loaded:
            print(f"Could not fetch {title}; skipping.")
            continue

        doc = loaded[0]
        page_title = doc.metadata.get("title") or title
        source = doc.metadata.get(
            "source", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        )
        path = DOCS_DIR / f"wikipedia_{slugify(page_title)}.md"
        path.write_text(
            f"---\ntitle: {page_title}\nsource: {source}\n"
            f"retrieved: {date.today().isoformat()}\n---\n\n{doc.page_content}\n",
            encoding="utf-8",
        )
        print(f"Wrote {path.relative_to(ROOT)} ({len(doc.page_content)} chars)")


if __name__ == "__main__":
    main()
