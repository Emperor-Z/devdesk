import os

import pytest

from devdesk.router_agent import root_agent
from devdesk.specialists.ares_agent import ares_agent
from devdesk.specialists.verisim_agent import verisim_agent


def test_specialists_construct_with_expected_tools():
    for agent, expected_tool_count in (
        (verisim_agent, 2),
        (ares_agent, 2),
    ):
        assert agent.name.endswith("_agent")
        assert len(agent.tools) == expected_tool_count


def test_router_has_one_agent_tool_per_specialist_plus_search_all():
    assert root_agent.name == "devdesk_router"
    assert len(root_agent.tools) == 3


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="requires a live GOOGLE_API_KEY")
async def test_router_end_to_end_smoke():
    from devdesk.cli import ask

    answer = await ask("What agents make up the Ares stack?")
    assert answer
