"""Ingest a project's real docs (markdown only) into its vector store.

No doc-generation fallback this phase: only content that actually exists in
the project's docs goes into the index, so nothing in the corpus is
model-synthesized. See context.md for why this was deferred.
"""

from __future__ import annotations

import time
from pathlib import Path

from devdesk.config import PROJECTS, ProjectConfig
from devdesk.rag.chunking import chunk_markdown
from devdesk.rag.vectorstore import VectorStore

_DOC_GLOBS = ("*.md", "*.MD")
_SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    ".chroma",
    # Tool caches ship their own boilerplate README.md — not project docs.
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}

# One embed_content call per file; free-tier Gemini embedding quota has
# been hit in practice well under its documented per-minute ceiling, so
# space calls out defensively rather than relying on retry alone.
_SECONDS_BETWEEN_FILES = 3.0


def _iter_doc_files(source_path: Path):
    for pattern in _DOC_GLOBS:
        for path in source_path.rglob(pattern):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            yield path


def ingest_project(project: ProjectConfig, store: VectorStore | None = None) -> int:
    """Chunk + embed + upsert every markdown doc for one project.

    Returns the number of source files ingested. Idempotent: re-running
    after edits upserts changed chunks, drops stale ones per file, and
    removes files that no longer exist.
    """
    if project.source_path is None:
        return 0

    store = store or VectorStore()
    count = 0
    ingested: list[str] = []
    for path in _iter_doc_files(project.source_path):
        if count > 0:
            time.sleep(_SECONDS_BETWEEN_FILES)
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_markdown(text)
        source_file = str(path.relative_to(project.source_path))
        store.upsert_source_file(project.name, project.collection_name, source_file, chunks)
        ingested.append(source_file)
        count += 1
    # Per-file upsert only cleans stale chunks inside files that still exist;
    # this drops files that were deleted (or newly skipped) since last run.
    store.prune_source_files(project.collection_name, keep=set(ingested))
    return count


def ingest_all(store: VectorStore | None = None) -> dict[str, int]:
    store = store or VectorStore()
    return {name: ingest_project(cfg, store) for name, cfg in PROJECTS.items()}


if __name__ == "__main__":
    results = ingest_all()
    for name, n in results.items():
        print(f"{name}: {n} file(s) ingested")
