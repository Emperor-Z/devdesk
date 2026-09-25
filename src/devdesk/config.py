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

# MiniLM's own hard cutoff is 256 tokens; every embedding backend is capped
# to the stricter of the two so chunk size never silently truncates.
MAX_CHUNK_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 35

MIN_QUERY_SIMILARITY = 0.35


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
