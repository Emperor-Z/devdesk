import asyncio
from types import SimpleNamespace

from devdesk.hardening.error_handling import GracefulDegradationPlugin
from devdesk.hardening.rate_limit import RateLimitPlugin


def _fake_callback_context(agent_name="ares_agent"):
    return SimpleNamespace(agent_name=agent_name, invocation_id="inv1")


def test_rate_limit_lets_first_call_through_without_waiting():
    clock = SimpleNamespace(now=1000.0)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    plugin = RateLimitPlugin(rpm=10, clock=lambda: clock.now, sleep=fake_sleep)

    asyncio.run(plugin.before_model_callback(callback_context=None, llm_request=None))
    assert sleeps == []


def test_rate_limit_blocks_second_call_within_interval():
    clock = SimpleNamespace(now=1000.0)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    plugin = RateLimitPlugin(rpm=10, clock=lambda: clock.now, sleep=fake_sleep)

    asyncio.run(plugin.before_model_callback(callback_context=None, llm_request=None))
    asyncio.run(plugin.before_model_callback(callback_context=None, llm_request=None))

    assert len(sleeps) == 1
    assert abs(sleeps[0] - 6.0) < 0.01  # 60/10 rpm = 6s interval


def test_rate_limit_does_not_block_after_interval_has_elapsed():
    clock = SimpleNamespace(now=1000.0)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    plugin = RateLimitPlugin(rpm=10, clock=lambda: clock.now, sleep=fake_sleep)

    asyncio.run(plugin.before_model_callback(callback_context=None, llm_request=None))
    clock.now += 10.0  # well past the 6s interval
    asyncio.run(plugin.before_model_callback(callback_context=None, llm_request=None))

    assert sleeps == []


def test_graceful_degradation_tool_error_returns_dict_not_raise():
    plugin = GracefulDegradationPlugin()
    tool = SimpleNamespace(name="search_docs")

    result = asyncio.run(
        plugin.on_tool_error_callback(
            tool=tool,
            tool_args={},
            tool_context=SimpleNamespace(),
            error=RuntimeError("503 UNAVAILABLE"),
        )
    )

    assert "search_docs" in result["error"]
    assert "RuntimeError" in result["error"]


def test_graceful_degradation_model_error_returns_llm_response_not_raise():
    plugin = GracefulDegradationPlugin()

    response = asyncio.run(
        plugin.on_model_error_callback(
            callback_context=_fake_callback_context("ares_agent"),
            llm_request=None,
            error=RuntimeError("429 RESOURCE_EXHAUSTED"),
        )
    )

    text = "".join(p.text or "" for p in response.content.parts)
    assert "ares_agent" in text
    assert response.error_code == "RuntimeError"
