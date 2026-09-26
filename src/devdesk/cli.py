"""One-shot CLI entrypoint: python -m devdesk.cli "question".

Doubles as the personal daily-use interface and the manual smoke-test path.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from google.adk.runners import InMemoryRunner
from google.genai import types

from devdesk.hardening.error_handling import GracefulDegradationPlugin
from devdesk.hardening.rate_limit import RateLimitPlugin
from devdesk.observability.logging import StructuredLoggingPlugin
from devdesk.observability.tracing import LangfuseTracingPlugin
from devdesk.router_agent import root_agent

_APP_NAME = "devdesk"
_USER_ID = "local"


@dataclass
class RunResult:
    answer: str
    # Tools the router itself called, in order (specialists appear under
    # their agent name via AgentTool). Specialist-internal calls run in
    # AgentTool's own sub-invocation and aren't surfaced here.
    router_tool_calls: list[str] = field(default_factory=list)


def build_runner() -> InMemoryRunner:
    """One runner (and one plugin set) per process: RateLimitPlugin's pacing
    state lives on the instance, so reusing it across questions is what
    keeps a batch of questions under the free-tier RPM ceiling."""
    return InMemoryRunner(
        agent=root_agent,
        app_name=_APP_NAME,
        plugins=[
            StructuredLoggingPlugin(),
            LangfuseTracingPlugin(),
            RateLimitPlugin(),
            GracefulDegradationPlugin(),
        ],
    )


async def run_question(runner: InMemoryRunner, question: str) -> RunResult:
    session = await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID
    )
    message = types.Content(role="user", parts=[types.Part(text=question)])

    result = RunResult(answer="")
    try:
        async for event in runner.run_async(
            user_id=_USER_ID, session_id=session.id, new_message=message
        ):
            if event.author == root_agent.name:
                result.router_tool_calls.extend(c.name for c in event.get_function_calls())
            if event.is_final_response() and event.content and event.content.parts:
                result.answer = "".join(p.text or "" for p in event.content.parts)
    finally:
        # Questions are one-shot; without this a long-running server (or a
        # big eval batch) accumulates every session in memory forever.
        await runner.session_service.delete_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session.id
        )
    return result


async def ask(question: str) -> str:
    return (await run_question(build_runner(), question)).answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m devdesk.cli "your question"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    answer = asyncio.run(ask(question))
    print(answer)


if __name__ == "__main__":
    main()
