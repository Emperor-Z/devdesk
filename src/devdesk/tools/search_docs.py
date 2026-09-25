"""Tool: search one project's indexed docs."""

from __future__ import annotations

from devdesk.config import PROJECTS
from devdesk.rag.vectorstore import VectorStore

_store = VectorStore()


def search_docs(project: str, query: str, k: int = 5) -> dict:
    """Search a single project's document index for passages relevant to `query`.

    Args:
        project: One of the configured project names (e.g. "verisim",
            "ares").
        query: The natural-language question or search text.
        k: Max number of passages to return.

    Returns:
        A dict with `hits`: a list of {text, citation, source_file,
        header_path, similarity} for every passage above the relevance
        cutoff, in descending similarity order. `citation` is the exact
        string to quote in your answer, e.g. "(architecture.md > Phase 3
        live execution)" — use it verbatim, don't reconstruct it from the
        other fields. An empty `hits` list means nothing in this project's
        docs was relevant enough — say so, don't guess.
    """
    cfg = PROJECTS.get(project)
    if cfg is None:
        return {"error": f"unknown project '{project}'", "hits": []}

    hits = _store.query(cfg.collection_name, query, k=k)
    result: dict = {
        "hits": [
            {
                "text": h.text,
                "citation": f"({h.source_file} > {h.header_path})",
                "source_file": h.source_file,
                "header_path": h.header_path,
                "similarity": round(h.similarity, 3),
            }
            for h in hits
        ]
    }
    return result
