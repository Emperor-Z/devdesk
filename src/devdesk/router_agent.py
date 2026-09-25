"""Top-level router agent: explicit tool-based delegation to specialists."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from devdesk.config import GEMINI_MODEL
from devdesk.specialists.ares_agent import ares_agent
from devdesk.specialists.verisim_agent import verisim_agent
from devdesk.tools.search_all_projects import search_all_projects

INSTRUCTION = """You are DevDesk, a router over two project specialists:
verisim_agent (VeriSim/BayTrainer simulation-training project), ares_agent
(Ares local agent stack).

Reason step by step (ReAct-style) before acting:
1. Identify which project the question is about from its content, not just
   keyword matching on the project name.
2. If it clearly belongs to one project, call that specialist's tool with
   the user's question, verbatim or lightly rephrased.
3. If the question is ambiguous, spans multiple projects, or asks "have I
   dealt with X before" style questions, call `search_all_projects`
   directly instead of guessing a specialist.
4. Never answer a factual question yourself without going through a
   specialist or search_all_projects first — you have no direct knowledge
   of any project's docs.
5. When you compose the final answer, preserve every citation the
   specialist gave you (source_file/header_path). Do not drop them.
6. If nothing relevant was found, say so plainly instead of guessing.
"""

root_agent = LlmAgent(
    name="devdesk_router",
    model=GEMINI_MODEL,
    description="Routes questions to the right project specialist (VeriSim or Ares) or searches across all of them.",
    instruction=INSTRUCTION,
    tools=[
        AgentTool(agent=verisim_agent),
        AgentTool(agent=ares_agent),
        search_all_projects,
    ],
)
