"""Ares (local agent stack) specialist agent."""

from __future__ import annotations

from google.adk.agents import LlmAgent

from devdesk.config import GEMINI_MODEL
from devdesk.tools.git_status_lookup import git_status_lookup
from devdesk.tools.search_docs import search_docs

INSTRUCTION = """You are the Ares specialist. Ares is a local, Ollama-backed
multi-agent stack (orchestrator/coder/thinker/runner) with a real local
repository at /home/z/ares, intentionally kept separate from "Aries God
Mode".

Rules:
1. Call `search_docs` with project="ares" before answering any factual
   question. Never answer from general knowledge about agent frameworks.
2. Every factual claim must end with that hit's `citation` string from the
   tool result, quoted verbatim — do not reconstruct or paraphrase it.
3. If `search_docs` returns no hits, say plainly that you don't have
   relevant Ares docs for that question — do not guess or generalize.
4. Use `git_status_lookup` only for "where did I leave off" / recent
   activity questions, not general architecture questions.
"""


def _ares_search_docs(query: str, k: int = 5) -> dict:
    return search_docs(project="ares", query=query, k=k)


def _ares_git_status() -> dict:
    return git_status_lookup(project="ares")


ares_agent = LlmAgent(
    name="ares_agent",
    model=GEMINI_MODEL,
    description="Answers questions about the Ares local multi-agent stack's architecture and docs.",
    instruction=INSTRUCTION,
    tools=[_ares_search_docs, _ares_git_status],
)
