"""Graceful degradation for model/tool errors, as an ADK plugin.

Closes the gap found while live-testing observability: a specialist's
model call failing (e.g. free-tier 503/429 flakiness) used to propagate
as an uncaught exception all the way up through the router's AgentTool
call, killing the whole invocation. ADK's plugin callbacks can return a
value in place of the error instead of letting it propagate — that's
the mechanism used here, not a new retry layer (see rate_limit.py /
llm_client.py for where retry actually lives).
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types


class GracefulDegradationPlugin(BasePlugin):
    """Turns an unhandled model/tool error into a normal, reportable result."""

    def __init__(self, name: str = "graceful_degradation") -> None:
        super().__init__(name)

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "error": (
                f"{tool.name} is temporarily unavailable "
                f"({type(error).__name__}); try again shortly."
            )
        }

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse:
        message = (
            f"{callback_context.agent_name} couldn't reach the model right now "
            f"({type(error).__name__}); please try again shortly."
        )
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=message)]),
            error_code=type(error).__name__,
            error_message=str(error),
        )
