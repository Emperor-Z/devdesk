from pathlib import Path

from devdesk.rag.chunking import Chunk, chunk_markdown, count_tokens
from devdesk.rag.vectorstore import VectorStore, chunk_id


def test_chunk_markdown_respects_token_cap():
    long_section = "word " * 2000
    text = f"# Title\n\n{long_section}"
    chunks = chunk_markdown(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 50


def test_chunk_markdown_keeps_short_section_whole():
    text = "# Title\n\nJust a short paragraph.\n\n## Sub\n\nAnother short one."
    chunks = chunk_markdown(text, max_tokens=220, overlap_tokens=35)
    assert len(chunks) == 2
    assert chunks[0].header_path == "Title"
    assert chunks[1].header_path == "Title > Sub"


def test_chunk_markdown_overlap_present():
    long_section = " ".join(f"word{i}" for i in range(400))
    text = f"# Title\n\n{long_section}"
    chunks = chunk_markdown(text, max_tokens=50, overlap_tokens=10)
    first_tail = chunks[0].text.split()[-5:]
    second_head = chunks[1].text.split()[:15]
    assert any(w in second_head for w in first_tail)


def _vs(fake_embedder, tmp_path: Path) -> VectorStore:
    return VectorStore(persist_dir=str(tmp_path), embedder=fake_embedder)


def test_upsert_and_query_ranks_relevant_higher(fake_embedder, tmp_path):
    store = _vs(fake_embedder, tmp_path)
    chunks = [
        Chunk(text="VeriSim uses Convex for the backend", header_path="Backend", index=0),
        Chunk(text="Ares runs a local Ollama backed agent stack", header_path="Ares", index=0),
    ]
    store.upsert_source_file("verisim", "test_collection", "doc.md", chunks)

    hits = store.query("test_collection", "Convex backend", k=5, min_similarity=0.0)
    assert hits[0].text.startswith("VeriSim")


def test_query_below_similarity_cutoff_returns_nothing(fake_embedder, tmp_path):
    store = _vs(fake_embedder, tmp_path)
    chunks = [Chunk(text="VeriSim uses Convex for the backend", header_path="Backend", index=0)]
    store.upsert_source_file("verisim", "test_collection", "doc.md", chunks)

    hits = store.query("test_collection", "completely unrelated space weather forecast", k=5, min_similarity=0.9)
    assert hits == []


def test_upsert_idempotent_diff_replaces_stale_chunks(fake_embedder, tmp_path):
    store = _vs(fake_embedder, tmp_path)
    v1 = [Chunk(text="version one content", header_path="A", index=0)]
    store.upsert_source_file("proj", "coll", "doc.md", v1)

    v2 = [Chunk(text="version two content", header_path="A", index=0)]
    store.upsert_source_file("proj", "coll", "doc.md", v2)

    collection = store._collection("coll")  # test-only introspection
    remaining = collection.get()
    assert remaining["documents"] == ["version two content"]


def test_chunk_id_is_stable_and_content_sensitive():
    c1 = Chunk(text="hello", header_path="A", index=0)
    c2 = Chunk(text="hello world", header_path="A", index=0)
    assert chunk_id("p", "f.md", c1) == chunk_id("p", "f.md", c1)
    assert chunk_id("p", "f.md", c1) != chunk_id("p", "f.md", c2)


def test_prune_source_files_drops_deleted_files_only(fake_embedder, tmp_path):
    store = _vs(fake_embedder, tmp_path)
    store.upsert_source_file("proj", "coll", "keep.md", [Chunk(text="kept", header_path="A", index=0)])
    store.upsert_source_file("proj", "coll", "gone.md", [Chunk(text="gone", header_path="A", index=0)])

    removed = store.prune_source_files("coll", keep={"keep.md"})

    assert removed == ["gone.md"]
    assert store._collection("coll").get()["documents"] == ["kept"]


def test_ingest_skips_tool_cache_dirs_and_prunes_them(fake_embedder, tmp_path):
    from devdesk.config import ProjectConfig
    from devdesk.rag import ingest

    src = tmp_path / "src"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "real.md").write_text("# Real\n\nActual project docs.")
    (src / ".pytest_cache").mkdir()
    (src / ".pytest_cache" / "README.md").write_text("# pytest cache directory #")

    store = VectorStore(persist_dir=str(tmp_path / "chroma"), embedder=fake_embedder)
    # Simulate an index built before the cache dir was skipped.
    store.upsert_source_file(
        "proj", "coll", ".pytest_cache/README.md", [Chunk(text="cache", header_path="x", index=0)]
    )

    n = ingest.ingest_project(ProjectConfig("proj", src, "coll"), store)

    assert n == 1
    sources = {m["source_file"] for m in store._collection("coll").get()["metadatas"]}
    assert sources == {"docs/real.md"}


def test_min_query_similarity_is_backend_specific_and_overridable(monkeypatch):
    from devdesk import config

    monkeypatch.delenv("MIN_QUERY_SIMILARITY", raising=False)
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "k")
    assert config.min_query_similarity() == 0.64
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    assert config.min_query_similarity() == 0.35
    monkeypatch.setenv("MIN_QUERY_SIMILARITY", "0.5")
    assert config.min_query_similarity() == 0.5
