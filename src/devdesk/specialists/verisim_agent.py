"""VeriSim (BayTrainer) specialist agent."""

from __future__ import annotations

from google.adk.agents import LlmAgent

from devdesk.config import GEMINI_MODEL
from devdesk.tools.git_status_lookup import git_status_lookup
from devdesk.tools.search_docs import search_docs

INSTRUCTION = """You are the VeriSim specialist. VeriSim (also referred to
as BayTrainer) is a simulation-training project with a real local
repository at /home/z/verisim.

Rules:
1. Call `search_docs` with project="verisim" before answering any factual
   question. Never answer from general knowledge about simulation training.
2. Every factual claim must end with that hit's `citation` string from the
   tool result, quoted verbatim — do not reconstruct or paraphrase it.
3. If `search_docs` returns no hits, say plainly that you don't have
   relevant VeriSim docs for that question — do not guess or generalize.
4. Use `git_status_lookup` only for "where did I leave off" / recent
   activity questions, not general architecture questions.
"""


def _verisim_search_docs(query: str, k: int = 5) -> dict:
    return search_docs(project="verisim", query=query, k=k)


def _verisim_git_status() -> dict:
    return git_status_lookup(project="verisim")


verisim_agent = LlmAgent(
    name="verisim_agent",
    model=GEMINI_MODEL,
    description="Answers questions about the VeriSim/BayTrainer simulation-training project's architecture and docs.",
    instruction=INSTRUCTION,
    tools=[_verisim_search_docs, _verisim_git_status],
)
