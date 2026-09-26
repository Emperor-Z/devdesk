"""Langfuse trace export as an ADK plugin.

Pushes the same before/after boundaries as StructuredLoggingPlugin
(logging.py) into a self-hosted Langfuse instance, so the actual trace UI
comes for free from existing infra rather than a new tracing stack.
No-op by construction when Langfuse isn't configured — this plugin must
never be able to break agent execution.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from devdesk import config

_SUMMARY_LIMIT = 300


def _truncate(value: Any) -> str:
    text = str(value)
    return text if len(text) <= _SUMMARY_LIMIT else text[:_SUMMARY_LIMIT] + "…"


def _content_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(p.text or "" for p in content.parts)


def _response_text(llm_response: LlmResponse) -> str:
    return _content_text(llm_response.content)


def _last_request_text(llm_request: LlmRequest) -> str:
    if not llm_request.contents:
        return ""
    return _content_text(llm_request.contents[-1])


class LangfuseTracingPlugin(BasePlugin):
    """Mirrors agent steps into Langfuse as a trace with span/generation children."""

    def __init__(self, name: str = "langfuse_tracing") -> None:
        super().__init__(name)
        self._enabled = config.use_langfuse()
        self._client = None
        if self._enabled:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
        self._spans: dict[str, Any] = {}
        # ADK constructs a fresh CallbackContext per callback invocation, so
        # id(callback_context) does not correlate before/after for one model
        # call. Model calls within an agent's turn run strictly sequentially,
        # so a per-invocation-id LIFO stack is safe instead.
        self._generations: dict[str, list[Any]] = {}

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> None:
        if not self._enabled:
            return
        self._client.trace(
            id=invocation_context.invocation_id,
            name="devdesk_invocation",
            session_id=invocation_context.session.id,
            input=_truncate(_content_text(user_message)),
        )

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> None:
        if not self._enabled:
            return
        span = self._client.span(
            trace_id=tool_context.invocation_id,
            name=tool.name,
            input=_truncate(tool_args),
        )
        self._spans[tool_context.function_call_id] = span

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> None:
        if not self._enabled:
            return
        span = self._spans.pop(tool_context.function_call_id, None)
        if span is not None:
            span.end(output=_truncate(result))

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> None:
        if not self._enabled:
            return
        span = self._spans.pop(tool_context.function_call_id, None)
        if span is not None:
            span.end(level="ERROR", status_message=_truncate(error))

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        if not self._enabled:
            return
        generation = self._client.generation(
            trace_id=callback_context.invocation_id,
            name=callback_context.agent_name,
            model=config.GEMINI_MODEL,
            input=_truncate(_last_request_text(llm_request)),
        )
        self._generations.setdefault(callback_context.invocation_id, []).append(generation)

    def _pop_generation(self, invocation_id: str) -> Any:
        stack = self._generations.get(invocation_id)
        return stack.pop() if stack else None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        if not self._enabled:
            return
        generation = self._pop_generation(callback_context.invocation_id)
        if generation is not None:
            generation.end(output=_truncate(_response_text(llm_response)))

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> None:
        if not self._enabled:
            return
        generation = self._pop_generation(callback_context.invocation_id)
        if generation is not None:
            generation.end(level="ERROR", status_message=_truncate(error))

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        if not self._enabled:
            return
        self._client.flush()
