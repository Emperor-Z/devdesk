import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import types

from devdesk.observability.logging import StructuredLoggingPlugin
from devdesk.observability.tracing import LangfuseTracingPlugin


def _fake_tool_context(function_call_id="fc1", invocation_id="inv1", agent_name="ares_agent"):
    return SimpleNamespace(
        function_call_id=function_call_id,
        invocation_id=invocation_id,
        agent_name=agent_name,
    )


def _fake_callback_context(invocation_id="inv1", agent_name="ares_agent"):
    return SimpleNamespace(invocation_id=invocation_id, agent_name=agent_name)


def _fake_invocation_context(invocation_id="inv1", session_id="sess1"):
    return SimpleNamespace(invocation_id=invocation_id, session=SimpleNamespace(id=session_id))


def _llm_response(text="an answer"):
    return SimpleNamespace(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _llm_request(text="a question"):
    content = types.Content(role="user", parts=[types.Part(text=text)])
    return SimpleNamespace(contents=[content])


# --- StructuredLoggingPlugin ---


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_tool_call_start_and_end_logged_with_latency(tmp_path):
    with patch("devdesk.observability.logging.LOG_DIR", tmp_path):
        plugin = StructuredLoggingPlugin()
        tool = SimpleNamespace(name="search_docs")
        tool_context = _fake_tool_context()

        asyncio.run(
            plugin.before_tool_callback(tool=tool, tool_args={"q": "x"}, tool_context=tool_context)
        )
        asyncio.run(
            plugin.after_tool_callback(
                tool=tool, tool_args={"q": "x"}, tool_context=tool_context, result={"hits": []}
            )
        )

    records = _read_jsonl(tmp_path / "devdesk.jsonl")
    assert [r["event"] for r in records] == ["tool_call_start", "tool_call_end"]
    end = records[1]
    assert end["tool"] == "search_docs"
    assert end["latency_ms"] >= 0


def test_tool_call_error_logged(tmp_path):
    with patch("devdesk.observability.logging.LOG_DIR", tmp_path):
        plugin = StructuredLoggingPlugin()
        tool = SimpleNamespace(name="search_docs")
        tool_context = _fake_tool_context()

        asyncio.run(
            plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=tool_context)
        )
        asyncio.run(
            plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=tool_context, error=RuntimeError("boom")
            )
        )

    records = _read_jsonl(tmp_path / "devdesk.jsonl")
    assert records[-1]["event"] == "tool_call_error"
    assert "boom" in records[-1]["error"]


def test_model_call_start_and_end_logged(tmp_path):
    with patch("devdesk.observability.logging.LOG_DIR", tmp_path):
        plugin = StructuredLoggingPlugin()
        callback_context = _fake_callback_context()

        asyncio.run(
            plugin.before_model_callback(
                callback_context=callback_context, llm_request=_llm_request()
            )
        )
        asyncio.run(
            plugin.after_model_callback(
                callback_context=callback_context, llm_response=_llm_response("hi")
            )
        )

    records = _read_jsonl(tmp_path / "devdesk.jsonl")
    assert [r["event"] for r in records] == ["model_call_start", "model_call_end"]
    assert records[1]["output_summary"] == "hi"


def test_run_start_and_end_logged(tmp_path):
    with patch("devdesk.observability.logging.LOG_DIR", tmp_path):
        plugin = StructuredLoggingPlugin()
        invocation_context = _fake_invocation_context()

        asyncio.run(plugin.before_run_callback(invocation_context=invocation_context))
        asyncio.run(plugin.after_run_callback(invocation_context=invocation_context))

    records = _read_jsonl(tmp_path / "devdesk.jsonl")
    assert [r["event"] for r in records] == ["run_start", "run_end"]
    assert records[0]["session_id"] == "sess1"


# --- LangfuseTracingPlugin ---


def test_langfuse_plugin_is_noop_without_keys():
    with patch("devdesk.config.use_langfuse", return_value=False):
        plugin = LangfuseTracingPlugin()
    assert plugin._client is None

    tool = SimpleNamespace(name="search_docs")
    tool_context = _fake_tool_context()
    # None of these should raise even though there's no client.
    asyncio.run(
        plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=tool_context)
    )
    asyncio.run(
        plugin.after_tool_callback(tool=tool, tool_args={}, tool_context=tool_context, result={})
    )
    asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=_fake_invocation_context(),
            user_message=types.Content(role="user", parts=[types.Part(text="hi")]),
        )
    )


def test_langfuse_plugin_creates_and_ends_span_and_generation():
    fake_client = MagicMock()
    fake_span = MagicMock()
    fake_generation = MagicMock()
    fake_client.span.return_value = fake_span
    fake_client.generation.return_value = fake_generation

    with patch("devdesk.config.use_langfuse", return_value=True), patch(
        "langfuse.Langfuse", return_value=fake_client
    ):
        plugin = LangfuseTracingPlugin()

    tool = SimpleNamespace(name="search_docs")
    tool_context = _fake_tool_context()
    asyncio.run(
        plugin.before_tool_callback(tool=tool, tool_args={"q": "x"}, tool_context=tool_context)
    )
    fake_client.span.assert_called_once()
    assert fake_client.span.call_args.kwargs["trace_id"] == "inv1"

    asyncio.run(
        plugin.after_tool_callback(
            tool=tool, tool_args={"q": "x"}, tool_context=tool_context, result={"hits": []}
        )
    )
    fake_span.end.assert_called_once()

    callback_context = _fake_callback_context()
    asyncio.run(
        plugin.before_model_callback(
            callback_context=callback_context, llm_request=_llm_request()
        )
    )
    fake_client.generation.assert_called_once()

    asyncio.run(
        plugin.after_model_callback(
            callback_context=callback_context, llm_response=_llm_response("hi")
        )
    )
    fake_generation.end.assert_called_once()
