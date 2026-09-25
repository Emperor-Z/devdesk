"""One-shot CLI entrypoint: python -m devdesk.cli "question".

Doubles as the personal daily-use interface and the manual smoke-test path.
"""

from __future__ import annotations

import asyncio
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from devdesk.router_agent import root_agent

_APP_NAME = "devdesk"
_USER_ID = "local"


async def ask(question: str) -> str:
    runner = InMemoryRunner(agent=root_agent, app_name=_APP_NAME)
    session = await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID
    )
    message = types.Content(role="user", parts=[types.Part(text=question)])

    final_text = ""
    async for event in runner.run_async(
        user_id=_USER_ID, session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)
    return final_text


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m devdesk.cli "your question"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    answer = asyncio.run(ask(question))
    print(answer)


if __name__ == "__main__":
    main()
