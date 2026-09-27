"""Project (tenant) registry and environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Source checkout by default; a non-editable install (the Docker image) lives
# in site-packages, so it sets DEVDESK_HOME to where data/ and logs/ are.
REPO_ROOT = Path(
    os.environ.get("DEVDESK_HOME") or Path(__file__).resolve().parents[2]
)
load_dotenv(REPO_ROOT / ".env")

CHROMA_DIR = REPO_ROOT / "data" / ".chroma"

# Free-tier limits decide this more than quality does. gemini-2.5-flash
# allows only 20 requests/day on this key (~4 questions at 5+ calls each);
# gemini-3.8-flash is 5 req/min and was 503-prone. gemini-3.5-flash-lite
# ran the full 16-query e2e eval with zero quota errors — see README
# "Evaluation" for its scores. Override via env to compare models.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.environ.get(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

LOG_DIR = REPO_ROOT / "logs"
# JSONL file sink next to stdout. On by default locally; the Docker image
# turns it off because Cloud Run's filesystem is RAM (the file would grow
# into the memory limit) and stdout already goes to Cloud Logging.
LOG_TO_FILE = os.environ.get("DEVDESK_LOG_FILE", "1") != "0"

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
    """One tenant. This is the whole onboarding surface: the specialist
    agent, its tools, and the router's routing guidance are all generated
    from these fields (see specialists/factory.py, router_agent.py)."""

    name: str
    source_path: Path | None
    collection_name: str
    display_name: str = ""
    # What the project is and what its indexed docs cover. The router
    # routes on this, so write it from the doc inventory: topics a user
    # would ask about, including the less obvious ones.
    description: str = ""


def _env_path(var: str, default: str | None) -> Path | None:
    raw = os.environ.get(var, default)
    if not raw:
        return None
    path = Path(raw).expanduser()
    # A path that doesn't exist here (e.g. inside the Cloud Run image) means
    # no local repo: the index still works, git_status_lookup is omitted.
    return path if path.exists() else None


PROJECTS: dict[str, ProjectConfig] = {
    "verisim": ProjectConfig(
        name="verisim",
        source_path=_env_path("VERISIM_SOURCE_PATH", "/home/z/verisim"),
        collection_name="devdesk_verisim",
        display_name="VeriSim (also called BayTrainer)",
        description=(
            "A clinical simulation-training game built on the AI Town codebase. "
            "Docs cover: its architecture (Convex backend, game engine and "
            "simulation loop, agent conversations and memories, client UI); the "
            "'Chest Pain, A&E Bay 3' scenario and its persona cards (patient, "
            "nurse, bystander, director); the consent-gate trust mechanic "
            "(state machine, classifier events, difficulty tuning, validation "
            "status); build notes and findings; running its LLM through Ollama "
            "on a Kaggle or Colab GPU; hosting on Fly.io; and the level editor."
        ),
    ),
    "ares": ProjectConfig(
        name="ares",
        source_path=_env_path("ARES_SOURCE_PATH", "/home/z/ares"),
        collection_name="devdesk_ares",
        display_name="Ares",
        description=(
            "A local, Ollama-backed multi-agent CLI/REPL (orchestrator plus "
            "coder, thinker and runner agents, Serena code navigation, mem0 "
            "long-term memory, slash commands, install and requirements). Docs "
            "also cover its independence plan from OpenJarvis: the runtime "
            "boundary, next steps, test plan, and a Phase 2 intelligence layer "
            "(Reflexion, Voyager skill library, ReWOO planning, Self-RAG, "
            "SWE-agent tools, hardware-aware model profiles)."
        ),
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
