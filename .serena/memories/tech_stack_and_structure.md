# Tech stack & structure

Python >=3.11, package under `src/devdesk/` (src layout, installed via
pyproject.toml / hatchling).

Stack (free-tier only, by explicit constraint):
- Agent framework: `google-adk`
- Vector store: `chromadb` (local)
- Embeddings: `sentence-transformers` (local, no paid API)
- LLM: Gemini free-tier API key
- Retry/backoff: `tenacity`
- Config: `pyyaml`
- Tests: `pytest` + `pytest-asyncio`
- Lint/type: `ruff`, `mypy`

Directory layout:
```
src/devdesk/
  router_agent.py          # ADK router, ReAct loop, delegates by project
  config.py                # project/tenant registry
  specialists/
    verisim_agent.py
    ares_agent.py
  tools/
    search_docs.py
    search_all_projects.py
    git_status_lookup.py
  rag/
    chunking.py
    embeddings.py
    vectorstore.py         # Chroma wrapper
    doc_generator.py       # fallback: synthesize docs from codebase
  observability/
    logging.py              # structured per-step logs
    tracing.py
  hardening/
    rate_limit.py
    retry.py
eval/
  queries.yaml               # 20-30 test queries + expected outcomes
  run_eval.py
  results/
tests/
  test_agents.py
  test_tools.py
  test_rag.py
deploy/
  cloudrun_service.yaml
  iam_setup.md
Dockerfile, docker-compose.yml, .env.example
```

Build order: scaffold (done) →
agents + tools → RAG pipeline → observability → hardening → eval harness →
deployment → README.
