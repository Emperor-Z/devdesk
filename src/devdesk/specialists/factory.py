"""Build a specialist agent for any configured tenant.

Every specialist is the same agent with a different project bound in, so
it's generated from `ProjectConfig` rather than hand-written per tenant.
Adding a tenant is a config entry plus an ingest run.
"""

from __future__ import annotations

from collections.abc import Callable

from google.adk.agents import LlmAgent

from devdesk.config import GEMINI_MODEL, PROJECTS, ProjectConfig
from devdesk.tools.git_status_lookup import git_status_lookup
from devdesk.tools.search_docs import search_docs

_INSTRUCTION = """You are the {display_name} specialist. {description}

Rules:
1. Call `{search_tool}` before answering any factual question. Never answer
   from general knowledge.
2. Every factual claim must end with that hit's `citation` string from the
   tool result, quoted verbatim — do not reconstruct or paraphrase it.
3. If `{search_tool}` returns no hits, say plainly that you don't have
   relevant {display_name} docs for that question — do not guess or
   generalize.
4. Keep the specifics. When a passage names concrete identifiers —
   functions, files, commands, config keys, model names — use them exactly
   as written instead of describing them in general terms. "The `migrate`
   command in `db/cli.py`" is an answer; "a migration step" is not.
5. Answer from the passage that most directly addresses the question. The
   section heading in each citation (after ">") tells you what a passage is
   about: for "how do I install it", an Install section beats an overview
   that happens to mention installation. When several hits each directly
   address it (the same topic covered in two sections), combine them —
   don't stop at the first. If no hit directly addresses the question,
   search again with a more specific query before answering.
{git_rule}"""

_GIT_RULE = """6. Use `{git_tool}` only for "where did I leave off" / recent activity
   questions, not general architecture questions.
"""


def _search_tool(cfg: ProjectConfig) -> Callable[..., dict]:
    def search(query: str, k: int = 5) -> dict:
        return search_docs(project=cfg.name, query=query, k=k)

    search.__name__ = f"search_{cfg.name}_docs"
    # ADK builds the tool declaration from the signature and docstring.
    search.__doc__ = f"""Search the {cfg.display_name} docs for passages relevant to `query`.

    Args:
        query: The natural-language question or search text.
        k: Max number of passages to return.

    Returns:
        A dict with `hits`: {{text, citation, source_file, header_path,
        similarity}} per passage, best first. Empty means nothing relevant.
    """
    return search


def _git_tool(cfg: ProjectConfig) -> Callable[[], dict]:
    def git_status() -> dict:
        return git_status_lookup(project=cfg.name)

    git_status.__name__ = f"{cfg.name}_git_status"
    git_status.__doc__ = f"""Recent commits and working-tree status of the local {cfg.display_name} repo."""
    return git_status


def make_specialist(cfg: ProjectConfig) -> LlmAgent:
    search = _search_tool(cfg)
    tools: list[Callable] = [search]
    git_rule = ""
    # Only offer git status where a repo actually exists — locally, not in
    # the Cloud Run image — so the agent can't call a tool that can't work.
    if cfg.source_path is not None:
        git = _git_tool(cfg)
        tools.append(git)
        git_rule = _GIT_RULE.format(git_tool=git.__name__)

    return LlmAgent(
        name=f"{cfg.name}_agent",
        model=GEMINI_MODEL,
        description=f"Answers questions about {cfg.display_name} from its docs.",
        instruction=_INSTRUCTION.format(
            display_name=cfg.display_name,
            description=cfg.description,
            search_tool=search.__name__,
            git_rule=git_rule,
        ),
        tools=tools,
    )


SPECIALISTS: dict[str, LlmAgent] = {name: make_specialist(cfg) for name, cfg in PROJECTS.items()}
