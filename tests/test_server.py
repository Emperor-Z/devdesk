import pytest
from fastapi.testclient import TestClient

from devdesk import config, server
from devdesk.cli import RunResult


@pytest.fixture
def client(monkeypatch):
    async def fake_run_question(runner, question):
        return RunResult(answer=f"echo: {question}", router_tool_calls=["ares_agent"])

    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(server, "build_runner", lambda: object())
    monkeypatch.setattr(server, "run_question", fake_run_question)
    monkeypatch.setattr(server, "index_chunk_counts", lambda: {"ares": 3, "verisim": 0})
    with TestClient(server.app) as c:
        yield c


def test_healthz_reports_index_size(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["index_chunks"] == {"ares": 3, "verisim": 0}


def test_ask_returns_answer_and_routing(client):
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "echo: hi", "router_tool_calls": ["ares_agent"]}


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_ask_rejects_empty_or_oversized_questions(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_server_refuses_to_start_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"), TestClient(server.app):
        pass


def test_server_refuses_to_start_with_an_empty_index(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(server, "index_chunk_counts", lambda: {"ares": 0, "verisim": 0})
    with pytest.raises(RuntimeError, match="ingest"), TestClient(server.app):
        pass
