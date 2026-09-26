# DevDesk — Context

## What this is
A multi-agent RAG support/assistant agent, built as a portfolio project for
a Google Forward Deployed Engineer (Applied AI) application, and designed
to be genuinely useful day to day as a personal knowledge assistant across
Arjun's own projects.

## Framing
Externally this is framed as a **generalizable customer-onboarding
pattern**: point the agent at a new codebase/doc set (a "tenant") and it
becomes a support agent for that project. The two tenants used to build
and eval it are Arjun's own public repos — VeriSim (BayTrainer) and Ares —
so the project doubles as a real personal tool, not just a demo.

## Tenant scope
The design started with a third tenant considered (a private client
project) but that was dropped entirely and scrubbed from the repo,
including git history, once it became clear that even a redacted
reference risked implying client confidentiality had been used as a
demo dataset. **Rule going forward: no private/client work of any kind
goes into this repo's tenant set, docs, examples, or commit history —
public repos only.** The repo (`Emperor-Z/devdesk`) was deleted and
recreated from scratch to guarantee this, since GitHub keeps merged PR
diffs viewable via internal refs even after a branch's history is
rewritten — rewriting `main` alone does not remove something from a
repo's history once it's been in a merged PR.

## Architecture
- **Router agent** — classifies which project a query is about (or routes
  to cross-project search), ReAct loop, explicit tool selection.
- **Specialist agents**, one per project:
  - `verisim_agent` — VeriSim/BayTrainer dissertation material
  - `ares_agent` — Ares agent-stack architecture docs
- **Tools**:
  1. `search_docs(project, query)` — RAG lookup scoped to one project's store
  2. `search_all_projects(query)` — cross-project search
  3. `git_status_lookup(project)` — recent commits/status for "where did I
     leave off" questions
- **Doc generation fallback**: `rag/doc_generator.py` stays an empty,
  deliberately deferred stub — nothing in the index is LLM-synthesized.
  New tenants are added by pointing `config.PROJECTS` at a source path
  with real docs and re-running `python -m devdesk.rag.ingest`.

## Constraints
- **Free-tier only**: local embeddings (sentence-transformers) as an
  offline fallback, `gemini-embedding-001` as the primary embedder, Chroma
  (local, free) as the vector store, Google ADK (free framework), Gemini
  free-tier API key for LLM calls, Docker locally, Cloud Run free tier for
  optional deploy.
- Python throughout; typed, docstring'd, pytest-covered.
- Public tenants only (see Tenant scope above).

## Build order
1. Scaffold repo structure — done
2. Agents + tools + minimal RAG (merged; `search_docs` needs a working
   vector store to mean anything, so these shipped together rather than
   RAG following as a separate phase) — done
3. Observability — done (structured JSON logging + Langfuse trace export,
   both as ADK plugins)
4. Production hardening (beyond the 429 retry + IPv4 fix already pulled
   forward — see below; also needs to fix the ungraceful-model-error gap
   found in this phase, see Status)
5. Eval harness scoring runner (2 seeded queries exist in
   `eval/queries.yaml`, harness itself not built yet)
6. Deployment (Docker, Cloud Run config, least-privilege IAM)
7. Final README pass

## Environment notes worth remembering
- **Python 3.14 is risky for `chromadb`** — build in a Python 3.12 venv
  (`uv venv -p 3.12`).
- **`torch` (via `sentence-transformers`) defaults to full CUDA wheels** on
  Linux — several GB unneeded on a machine with no GPU. Pinned to the
  CPU-only index in `pyproject.toml` (`[tool.uv.sources]` /
  `[[tool.uv.index]]`).
- **This machine's IPv6 route to Google's API is broken/blackholed** —
  every connection attempt burns 60-130s on a dead SYN handshake before
  falling back to IPv4. Fixed with an IPv4-only `socket.getaddrinfo`
  monkeypatch in `llm_client.py` (applies globally on import). If
  something looks hung talking to an external API on this machine, check
  `ss -tp` for `SYN-SENT` before assuming a code bug.
- **`gemini-embedding-001` free-tier quota gets hit well under its
  documented per-minute ceiling** in practice — `ingest.py` spaces
  per-file embed calls 3s apart, and `llm_client.py`'s retry is tuned to 5
  attempts / up to 30s backoff.
- **`gemini-3.8-flash` (current GA default in AI Studio) returned repeated
  `503 UNAVAILABLE` ("high demand")** from this free-tier key at build
  time; `gemini-2.5-flash` was more reliable and is the current default
  (`GEMINI_MODEL` env-overridable). Both models showed intermittent 503s
  on repeat testing — this reads as genuine external API overload, not a
  model-specific or code issue. Don't chase it by retrying in a loop.
- `.env` is loaded via `python-dotenv` in `config.py`, not shell-sourced —
  useful if a key ever arrives as pasted/external content again, since
  shell-sourcing that is (correctly) blocked by the permission classifier.
- **`gemini-3.8-flash`'s free tier is only 5 requests/minute** — far too
  low for this router+specialist pattern (5+ model calls per question).
  This was silently overriding the code default because the `.env` the
  key arrived in had the old default baked in; `.env` isn't tracked by
  git so this kind of drift from `.env.example` won't show up in a diff —
  worth checking `.env` directly, not just `.env.example`, if a run
  behaves like it's on a different model than expected.
- **ADK constructs a fresh `CallbackContext` per callback invocation** —
  `id(callback_context)` does not correlate a model call's
  `before_model_callback`/`after_model_callback` pair. Every latency read
  as exactly `0.0` until checked against a live run. Fixed in both
  observability plugins with a per-`invocation_id` LIFO stack instead
  (safe because one agent's model calls run strictly sequentially).
  `ToolContext.function_call_id` doesn't have this problem — it's a real
  per-call id.

## Status
Repo rebuilt clean after the tenant-scope decision above. Git history is
now a small number of fresh commits with no client references anywhere,
ever. Current state: scaffold, router + 2 specialist agents, RAG pipeline
(header-aware chunking capped at 220 MiniLM-tokenizer-counted tokens,
Gemini-embedding-001 primary / MiniLM offline fallback, Chroma vector
store with content-hash idempotent upsert and a 0.35 similarity cutoff,
pre-formatted `citation` string per hit), README, all committed to `main`
on a fresh private GitHub remote (`Emperor-Z/devdesk`) with branch
protection re-applied (PR required incl. for admins, no force-push, no
deletions).

Verified: real ingest against both tenants with live Gemini embeddings,
retrieval returns correctly cited hits, 15 unit tests pass, lint clean,
and one manual live router→specialist round trip succeeded (before the
rebuild, same code path — behavior is unchanged) showing correct
delegation and citations.

**Phase 3 (observability) done, on branch `feat/observability`**:
`StructuredLoggingPlugin` (JSON lines to stdout + `logs/devdesk.jsonl`)
and `LangfuseTracingPlugin` (mirrors the same steps into a self-hosted
Langfuse instance — no-op without keys), both as ADK `BasePlugin`
subclasses wired into `cli.py`'s `InMemoryRunner(plugins=[...])`. 20 unit
tests pass (6 new), lint clean. Verified against a real live run: latency
correlation bug found and fixed (see Environment notes), structured logs
correctly captured both a successful call and a real `503` error path end
to end. Langfuse UI verification itself still pending — needs the user to
create a "DevDesk" project at their self-hosted instance
(`localhost:3000`) and hand over the key pair.

**Real gap found during this phase's live testing, not yet fixed**: when
a specialist's model call fails (e.g. the ongoing external 503
flakiness), the exception propagates all the way up through the router's
`AgentTool` call as an uncaught exception, killing the whole CLI
invocation — there's no graceful degradation ("the Ares specialist is
temporarily unavailable, try again") the way `search_docs`'s empty-hits
path or `git_status_lookup`'s no-repo-configured path already handle
gracefully for their own failure modes. This belongs in the hardening
phase.

Not yet done: full hardening (the gap above, plus rate limiting beyond
the existing 429 retry), eval harness runner, deployment.
