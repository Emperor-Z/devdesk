"""Project (tenant) registry and environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

CHROMA_DIR = REPO_ROOT / "data" / ".chroma"

# gemini-3.8-flash (current GA default in AI Studio) returned repeated 503
# "high demand" errors from this free-tier key at build time; gemini-2.5-flash
# was reliable. Defaulting to the reliable one — override via env if 3.8
# stabilizes for you.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.environ.get(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

LOG_DIR = REPO_ROOT / "logs"

# MiniLM's own hard cutoff is 256 tokens; every embedding backend is capped
# to the stricter of the two so chunk size never silently truncates.
MAX_CHUNK_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 35

# Relevance cutoff for a retrieved chunk, per embedding backend — the two
# have very different similarity floors. Measured with `python
# eval/run_eval.py --mode retrieval` on gemini-embedding-001: relevant hits
# bottomed out at 0.661, out-of-scope queries topped out at 0.621, so 0.64
# splits the gap (a narrow margin — re-run the eval if the corpus or
# embedding model changes). MiniLM's 0.35 predates the eval harness and
# hasn't been re-measured. MIN_QUERY_SIMILARITY overrides both.
_MIN_QUERY_SIMILARITY_BY_BACKEND = {"gemini": 0.64, "minilm": 0.35}


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    source_path: Path | None
    collection_name: str


def _env_path(var: str, default: str | None) -> Path | None:
    raw = os.environ.get(var, default)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


PROJECTS: dict[str, ProjectConfig] = {
    "verisim": ProjectConfig(
        name="verisim",
        source_path=_env_path("VERISIM_SOURCE_PATH", "/home/z/verisim"),
        collection_name="devdesk_verisim",
    ),
    "ares": ProjectConfig(
        name="ares",
        source_path=_env_path("ARES_SOURCE_PATH", "/home/z/ares"),
        collection_name="devdesk_ares",
    ),
}


def use_gemini_embeddings() -> bool:
    return bool(GOOGLE_API_KEY)


def min_query_similarity() -> float:
    override = os.environ.get("MIN_QUERY_SIMILARITY")
    if override:
        return float(override)
    backend = "gemini" if use_gemini_embeddings() else "minilm"
    return _MIN_QUERY_SIMILARITY_BY_BACKEND[backend]


def use_langfuse() -> bool:
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
