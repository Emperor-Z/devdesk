"""Tool: search across every configured project's docs at once."""

from __future__ import annotations

from devdesk.citations import record_hits
from devdesk.config import PROJECTS
from devdesk.rag.vectorstore import VectorStore

_store = VectorStore()


def search_all_projects(query: str, k_per_project: int = 3) -> dict:
    """Search every project's document index for passages relevant to `query`.

    Use this when a question doesn't clearly belong to one project, or when
    checking whether something has "come up before" across projects.

    Args:
        query: The natural-language question or search text.
        k_per_project: Max passages to return per project before merging.

    Returns:
        A dict with `hits`: a list of {project, text, citation, source_file,
        header_path, similarity}, merged across all projects and sorted by
        similarity descending. `citation` is the exact string to quote in
        your answer, e.g. "(devdesk_ares: README.md > Architecture)" — use
        it verbatim. An empty list means nothing relevant was found
        anywhere — say so, don't guess.
    """
    merged: list[dict] = []
    for name, cfg in PROJECTS.items():
        for h in _store.query(cfg.collection_name, query, k=k_per_project):
            merged.append(
                {
                    "project": name,
                    "text": h.text,
                    "citation": f"({name}: {h.source_file} > {h.header_path})",
                    "source_file": h.source_file,
                    "header_path": h.header_path,
                    "similarity": round(h.similarity, 3),
                }
            )
    merged.sort(key=lambda h: h["similarity"], reverse=True)
    record_hits(merged)
    return {"hits": merged}
