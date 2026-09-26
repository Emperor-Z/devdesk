"""Structured per-step logging as an ADK plugin.

Registered once on the Runner (see cli.py), this applies uniformly to the
router and every specialist without touching agent code — ADK's plugin
callbacks fire at run/model/tool boundaries regardless of which agent is
executing.
"""

from __future__ import annotations

import json
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from devdesk.config import LOG_DIR

_SUMMARY_LIMIT = 300


def _truncate(value: Any) -> str:
    text = str(value)
    return text if len(text) <= _SUMMARY_LIMIT else text[:_SUMMARY_LIMIT] + "…"


def _response_text(llm_response: LlmResponse) -> str:
    if not llm_response.content or not llm_response.content.parts:
        return ""
    return "".join(p.text or "" for p in llm_response.content.parts)


class StructuredLoggingPlugin(BasePlugin):
    """Emits one JSON line per agent step: run/model/tool start, end, error."""

    def __init__(self, name: str = "structured_logging") -> None:
        super().__init__(name)
        self._starts: dict[object, float] = {}
        # ADK constructs a fresh CallbackContext per callback invocation, so
        # id(callback_context) does NOT correlate before/after for a given
        # model call. Model calls within one agent's turn run strictly
        # sequentially, so a per-invocation-id LIFO stack is safe instead.
        self._model_starts: dict[str, list[float]] = {}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = LOG_DIR / "devdesk.jsonl"

    def _emit(self, record: dict) -> None:
        line = json.dumps({"ts": time.time(), **record}, default=str)
        print(line)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._starts[invocation_context.invocation_id] = time.monotonic()
        self._emit(
            {
                "event": "run_start",
                "invocation_id": invocation_context.invocation_id,
                "session_id": invocation_context.session.id,
            }
        )

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        start = self._starts.pop(invocation_context.invocation_id, time.monotonic())
        self._emit(
            {
                "event": "run_end",
                "invocation_id": invocation_context.invocation_id,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
            }
        )

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        self._model_starts.setdefault(callback_context.invocation_id, []).append(
            time.monotonic()
        )
        self._emit(
            {
                "event": "model_call_start",
                "invocation_id": callback_context.invocation_id,
                "agent_name": callback_context.agent_name,
            }
        )

    def _pop_model_start(self, invocation_id: str) -> float:
        stack = self._model_starts.get(invocation_id)
        if stack:
            return stack.pop()
        return time.monotonic()

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        start = self._pop_model_start(callback_context.invocation_id)
        self._emit(
            {
                "event": "model_call_end",
                "invocation_id": callback_context.invocation_id,
                "agent_name": callback_context.agent_name,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "output_summary": _truncate(_response_text(llm_response)),
            }
        )

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> None:
        start = self._pop_model_start(callback_context.invocation_id)
        self._emit(
            {
                "event": "model_call_error",
                "invocation_id": callback_context.invocation_id,
                "agent_name": callback_context.agent_name,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "error": _truncate(error),
            }
        )

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> None:
        self._starts[tool_context.function_call_id] = time.monotonic()
        self._emit(
            {
                "event": "tool_call_start",
                "invocation_id": tool_context.invocation_id,
                "agent_name": tool_context.agent_name,
                "tool": tool.name,
                "args_summary": _truncate(tool_args),
            }
        )

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> None:
        start = self._starts.pop(tool_context.function_call_id, time.monotonic())
        self._emit(
            {
                "event": "tool_call_end",
                "invocation_id": tool_context.invocation_id,
                "agent_name": tool_context.agent_name,
                "tool": tool.name,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "result_summary": _truncate(result),
            }
        )

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> None:
        start = self._starts.pop(tool_context.function_call_id, time.monotonic())
        self._emit(
            {
                "event": "tool_call_error",
                "invocation_id": tool_context.invocation_id,
                "agent_name": tool_context.agent_name,
                "tool": tool.name,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "error": _truncate(error),
            }
        )
