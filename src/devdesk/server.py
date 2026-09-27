"""HTTP entrypoint — the same command locally and in the Cloud Run image:

    devdesk-serve            # listens on $PORT (default 8080)

One process-wide runner, so RateLimitPlugin paces every request together —
which is also why the Cloud Run service is pinned to a single instance
(see deploy/cloudrun_service.yaml): the free-tier quota is per project, not
per instance. No auth here on purpose: Cloud Run IAM (no allUsers invoker)
is the access boundary.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from devdesk import config
from devdesk.cli import build_runner, run_question
from devdesk.rag.vectorstore import VectorStore

_MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    answer: str
    router_tool_calls: list[str]


def index_chunk_counts() -> dict[str, int]:
    store = VectorStore()
    return {name: store.count(cfg.collection_name) for name, cfg in config.PROJECTS.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The baked-in index was embedded with Gemini; querying it with the
    # MiniLM fallback would fail on a dimension mismatch at the first
    # request. Refuse to start instead.
    if not config.use_gemini_embeddings():
        raise RuntimeError("GOOGLE_API_KEY is not set; refusing to start")
    # An empty index still "works" — every answer is just "no relevant
    # docs". Fail loudly instead, locally and in the image alike.
    app.state.index_chunks = index_chunk_counts()
    if not any(app.state.index_chunks.values()):
        raise RuntimeError(
            f"index at {config.CHROMA_DIR} is empty; run `python -m devdesk.rag.ingest`"
        )
    app.state.runner = build_runner()
    yield


app = FastAPI(title="DevDesk", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "model": config.GEMINI_MODEL,
        "index_chunks": app.state.index_chunks,
    }


@app.post("/ask")
async def ask(req: AskRequest) -> AskResponse:
    result = await run_question(app.state.runner, req.question)
    return AskResponse(answer=result.answer, router_tool_calls=result.router_tool_calls)


def main() -> None:
    import os

    import uvicorn

    # Cloud Run injects PORT; locally it defaults to 8080. Binding 0.0.0.0
    # is what a container needs; set HOST=127.0.0.1 to keep a local run private.
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
    )
