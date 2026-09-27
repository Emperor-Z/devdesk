"""HTTP entrypoint for container / Cloud Run deploys.

    uvicorn devdesk.server:app --port 8080

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

_MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    answer: str
    router_tool_calls: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The baked-in index was embedded with Gemini; querying it with the
    # MiniLM fallback would fail on a dimension mismatch at the first
    # request. Refuse to start instead.
    if not config.use_gemini_embeddings():
        raise RuntimeError("GOOGLE_API_KEY is not set; refusing to start")
    app.state.runner = build_runner()
    yield


app = FastAPI(title="DevDesk", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "model": config.GEMINI_MODEL}


@app.post("/ask")
async def ask(req: AskRequest) -> AskResponse:
    result = await run_question(app.state.runner, req.question)
    return AskResponse(answer=result.answer, router_tool_calls=result.router_tool_calls)
