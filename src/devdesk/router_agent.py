"""Top-level router agent: explicit tool-based delegation to specialists.

The tenant list, routing guidance and tools are all generated from
`config.PROJECTS`, so the router never needs editing to add a tenant.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from devdesk.config import GEMINI_MODEL, PROJECTS
from devdesk.specialists.factory import SPECIALISTS
from devdesk.tools.search_all_projects import search_all_projects

_INSTRUCTION = """You are DevDesk, a router over these project specialists:

{tenants}

Reason step by step (ReAct-style) before acting:
1. Identify which project the question is about from its content — match
   the topics it mentions against each project's description above, not
   just whether a project's name appears. Technologies can be shared
   across projects (several run models locally through Ollama), so decide
   on the most specific topic in the question, not a shared tool name.
2. If it clearly belongs to one project, call that specialist's tool with
   the user's question, verbatim or lightly rephrased.
3. If the question genuinely spans several projects, or asks "have I dealt
   with X before" style questions, or matches no description, call
   `search_all_projects` directly instead of guessing a specialist.
4. Never answer a factual question yourself without going through a
   specialist or search_all_projects first — you have no direct knowledge
   of any project's docs.
5. When you compose the final answer, preserve every citation the
   specialist gave you (source_file/header_path). Do not drop them.
6. If nothing relevant was found, say so plainly instead of guessing.
"""


def _tenant_lines() -> str:
    return "\n".join(
        f"- {SPECIALISTS[name].name} — {cfg.display_name}: {cfg.description}"
        for name, cfg in PROJECTS.items()
    )


root_agent = LlmAgent(
    name="devdesk_router",
    model=GEMINI_MODEL,
    description="Routes questions to the right project specialist or searches across all of them.",
    instruction=_INSTRUCTION.format(tenants=_tenant_lines()),
    tools=[*(AgentTool(agent=a) for a in SPECIALISTS.values()), search_all_projects],
)
