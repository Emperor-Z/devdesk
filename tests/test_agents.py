import os
from pathlib import Path

import pytest

from devdesk.config import PROJECTS, ProjectConfig
from devdesk.router_agent import root_agent
from devdesk.specialists.factory import SPECIALISTS, make_specialist


def _tool_names(agent):
    return [getattr(t, "__name__", getattr(t, "name", None)) for t in agent.tools]


def test_one_specialist_per_configured_tenant():
    assert set(SPECIALISTS) == set(PROJECTS)
    for name, agent in SPECIALISTS.items():
        assert agent.name == f"{name}_agent"
        assert f"search_{name}_docs" in _tool_names(agent)


def test_router_has_one_agent_tool_per_specialist_plus_search_all():
    assert root_agent.name == "devdesk_router"
    assert len(root_agent.tools) == len(PROJECTS) + 1


def test_router_instruction_describes_every_tenant():
    for cfg in PROJECTS.values():
        assert cfg.description in root_agent.instruction


def test_git_tool_only_offered_when_a_local_repo_exists(tmp_path: Path):
    local = make_specialist(ProjectConfig("demo", tmp_path, "c", "Demo", "A demo."))
    cloud = make_specialist(ProjectConfig("demo", None, "c", "Demo", "A demo."))
    assert _tool_names(local) == ["search_demo_docs", "demo_git_status"]
    assert _tool_names(cloud) == ["search_demo_docs"]
    assert "demo_git_status" not in cloud.instruction


def test_generated_search_tool_is_bound_to_its_tenant(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "devdesk.specialists.factory.search_docs",
        lambda project, query, k: calls.append((project, query, k)) or {"hits": []},
    )
    agent = make_specialist(ProjectConfig("demo", None, "c", "Demo", "A demo."))
    agent.tools[0]("what is it?")
    assert calls == [("demo", "what is it?", 5)]


def test_prompts_contain_no_machine_specific_paths():
    for agent in [root_agent, *SPECIALISTS.values()]:
        assert "/home/" not in agent.instruction


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="requires a live GOOGLE_API_KEY")
async def test_router_end_to_end_smoke():
    from devdesk.cli import ask

    answer = await ask("What agents make up the Ares stack?")
    assert answer
