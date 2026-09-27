import subprocess
from pathlib import Path
from unittest.mock import patch

from devdesk.config import ProjectConfig
from devdesk.rag.chunking import Chunk
from devdesk.rag.vectorstore import VectorStore
from devdesk.tools.git_status_lookup import git_status_lookup
from devdesk.tools.search_docs import search_docs


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_git_status_lookup_real_repo(tmp_path):
    _init_git_repo(tmp_path)
    fake_cfg = ProjectConfig(name="fake", source_path=tmp_path, collection_name="c")
    with patch("devdesk.tools.git_status_lookup.PROJECTS", {"fake": fake_cfg}):
        result = git_status_lookup("fake")
    assert result["status"] == "ok"
    assert "init" in result["recent_commits"]
    assert result["working_tree"] == "(clean)"


def test_git_status_lookup_no_local_path_returns_clean_message():
    fake_cfg = ProjectConfig(name="no-repo-project", source_path=None, collection_name="c")
    with patch("devdesk.tools.git_status_lookup.PROJECTS", {"no-repo-project": fake_cfg}):
        result = git_status_lookup("no-repo-project")
    assert "no local repo configured" in result["status"]


def test_git_status_lookup_unknown_project():
    result = git_status_lookup("does-not-exist")
    assert "unknown project" in result["status"]


def test_search_docs_returns_citation_metadata(fake_embedder, tmp_path):
    store = VectorStore(persist_dir=str(tmp_path), embedder=fake_embedder)
    store.upsert_source_file(
        "verisim",
        "devdesk_verisim",
        "ARCHITECTURE.md",
        [Chunk(text="VeriSim uses Convex for the backend", header_path="Backend", index=0)],
    )
    fake_cfg = ProjectConfig(name="verisim", source_path=tmp_path, collection_name="devdesk_verisim")
    with patch("devdesk.tools.search_docs.PROJECTS", {"verisim": fake_cfg}), \
         patch("devdesk.tools.search_docs._store", store):
        result = search_docs("verisim", "Convex backend", k=3)
    assert result["hits"]
    hit = result["hits"][0]
    assert hit["source_file"] == "ARCHITECTURE.md"
    assert hit["header_path"] == "Backend"


def test_search_docs_unknown_project_returns_error():
    result = search_docs("nope", "anything")
    assert result["hits"] == []
    assert "error" in result


def test_search_docs_records_hits_for_citation_verification(fake_embedder, tmp_path):
    from devdesk.citations import start_request

    store = VectorStore(persist_dir=str(tmp_path), embedder=fake_embedder)
    store.upsert_source_file(
        "verisim", "coll", "ARCHITECTURE.md",
        [Chunk(text="VeriSim uses Convex for the backend", header_path="Backend", index=0)],
    )
    fake_cfg = ProjectConfig(name="verisim", source_path=tmp_path, collection_name="coll")
    record = start_request()
    with patch("devdesk.tools.search_docs.PROJECTS", {"verisim": fake_cfg}), \
         patch("devdesk.tools.search_docs._store", store):
        search_docs("verisim", "Convex backend", k=3)
    assert record == {("ARCHITECTURE.md", "Backend")}
