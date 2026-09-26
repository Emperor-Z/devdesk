"""Chroma-backed vector store: one collection per project.

Idempotent ingestion: chunk IDs are content hashes, so re-running ingest
after a doc edit upserts only the changed chunks and deletes ones that no
longer exist for that source file, and `prune_source_files` drops files
that were deleted outright — no `force` flag, no stale data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb

from devdesk import config
from devdesk.rag.chunking import Chunk
from devdesk.rag.embeddings import Embedder, get_embedder


def chunk_id(project: str, source_file: str, chunk: Chunk) -> str:
    raw = f"{project}|{source_file}|{chunk.header_path}|{chunk.index}|{chunk.text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SearchHit:
    text: str
    source_file: str
    header_path: str
    similarity: float


class VectorStore:
    def __init__(self, persist_dir: str | None = None, embedder: Embedder | None = None) -> None:
        path = persist_dir or str(config.CHROMA_DIR)
        self._client = chromadb.PersistentClient(path=path)
        self._embedder = embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embedder = self._embedder or get_embedder()
        return embedder.embed_texts(texts)

    def _collection(self, project_collection: str):
        return self._client.get_or_create_collection(
            name=project_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_source_file(
        self, project: str, collection_name: str, source_file: str, chunks: list[Chunk]
    ) -> None:
        collection = self._collection(collection_name)
        existing = collection.get(where={"source_file": source_file})
        existing_ids = set(existing.get("ids", []))

        new_ids = [chunk_id(project, source_file, c) for c in chunks]
        if chunks:
            embeddings = self._embed([c.text for c in chunks])
            collection.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "project": project,
                        "source_file": source_file,
                        "header_path": c.header_path,
                    }
                    for c in chunks
                ],
            )

        stale_ids = existing_ids - set(new_ids)
        if stale_ids:
            collection.delete(ids=list(stale_ids))

    def prune_source_files(self, collection_name: str, keep: set[str]) -> list[str]:
        """Delete every chunk whose source_file isn't in `keep`; returns the
        removed source files."""
        collection = self._collection(collection_name)
        existing = collection.get(include=["metadatas"])
        stale_ids: list[str] = []
        removed: set[str] = set()
        for chunk_id_, meta in zip(existing.get("ids", []), existing.get("metadatas", [])):
            source_file = (meta or {}).get("source_file", "")
            if source_file not in keep:
                stale_ids.append(chunk_id_)
                removed.add(source_file)
        if stale_ids:
            collection.delete(ids=stale_ids)
        return sorted(removed)

    def query(
        self,
        collection_name: str,
        text: str,
        k: int = 5,
        min_similarity: float | None = None,
    ) -> list[SearchHit]:
        if min_similarity is None:
            min_similarity = config.min_query_similarity()
        collection = self._collection(collection_name)
        if collection.count() == 0:
            return []
        query_embedding = self._embed([text])[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=k)

        hits: list[SearchHit] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, distance in zip(docs, metas, distances):
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            hits.append(
                SearchHit(
                    text=doc,
                    source_file=meta.get("source_file", ""),
                    header_path=meta.get("header_path", ""),
                    similarity=similarity,
                )
            )
        return hits
