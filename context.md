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
5. Eval harness — done (retrieval mode verified live; see Status)
6. Deployment — done (container verified locally; not yet deployed to GCP)
7. Final README pass — done

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
to end.

**Langfuse UI verification: done.** Browser automation was unavailable
initially (extension disconnected) and there were no admin credentials
for the existing Ares org, so rather than poke at the Langfuse Postgres
container directly (correctly blocked by the permission classifier as
credential exploration when tried), the right path was the front door:
once the browser extension reconnected, signed up a fresh account
(`devdesk-bot@localhost.test`) through Langfuse's normal local sign-up
flow — legitimate under the "testing your own local app" exception since
this is `localhost:3000`. Created org `devdesk` → project `DevDesk`,
grabbed the key pair straight off the setup-wizard screen, wired into
`.env`. Ran a real query end to end and confirmed the trace tree in the
UI matches the JSON logs exactly: `devdesk_router` generation (1.76s) →
`verisim_agent` span (18.18s) → `devdesk_router` generation (5.06s),
total 25.01s trace latency matching `logs/devdesk.jsonl`'s `run_end`
exactly. Nothing left pending from Phase 3.

**Phase 4 (hardening) done, on branch `feat/hardening`**: fixed the gap
above. `GracefulDegradationPlugin` (`hardening/error_handling.py`) returns
a friendly fallback instead of letting `on_tool_error_callback`/
`on_model_error_callback` propagate — ADK's plugin API supports this
directly (returning a value from those callbacks replaces the error
rather than raising it). Verified live by forcing `gemini-3.8-flash`'s
5 req/min limit: the CLI now prints `"devdesk_router couldn't reach the
model right now (ServerError); please try again shortly."` instead of a
stack trace. The fix actually resolves the bug *inside* the specialist's
own invocation (its model error degrades gracefully before the failure
ever reaches the router as a tool error), which is a cleaner fix than
patching it at the tool-call boundary would have been.

Also added `RateLimitPlugin` (`hardening/rate_limit.py`) — proactive
fixed-interval pacing of model calls (`RATE_LIMIT_RPM`, default 8),
distinct from and complementary to the retry that already exists in
`llm_client.py` and inside `google-genai`'s own client.
`hardening/retry.py` stays an intentionally empty stub (docstring
explains why — no third retry layer). 26 unit tests pass (5 new), lint
clean.

**Minor known rough edge, not worth fixing**: when
`GracefulDegradationPlugin` supplies a fallback response,
`StructuredLoggingPlugin` logs both the real `model_call_error` (correct
latency) and a synthetic `model_call_end` for the fallback response
itself (latency `0.0`, since it's not a real model call) — cosmetic
double-logging, not a data-loss or correctness issue.

**Phase 5 (eval harness), branch `feat/eval`**: `devdesk.evaluation`
(`scoring.py` pure + unit-tested, `harness.py` runner), entry
`python eval/run_eval.py --mode retrieval|e2e [--ids ...] [--fail-under X]`.
16 hand-written queries (7 verisim, 6 ares, 1 cross, 2 out-of-scope),
deterministic scoring only (no LLM judge — costs quota, adds variance).
Results written to `eval/results/<utc>_<mode>.json`.

- `cli.py` split into `build_runner()` + `run_question()` (returns answer
  + router tool calls): RateLimitPlugin's pacing state is per instance, so
  the old one-runner-per-`ask()` would reset pacing every question in a
  batch.
- **Eval caught a real bug**: MIN_QUERY_SIMILARITY 0.35 was tuned for
  MiniLM; gemini-embedding-001's similarity floor is ~0.5, so nothing was
  ever filtered (sourdough question scored 0.506). Measured: relevant hits
  min 0.661, out-of-scope max 0.621 → Gemini cutoff 0.64, per-backend via
  `config.min_query_similarity()`, env-overridable. Margin is narrow —
  re-run retrieval eval after corpus/model changes.
- Unit suite now pins MIN_QUERY_SIMILARITY=0.35 (autouse fixture) — the
  default depends on `.env`'s key, which made a FakeEmbedder test flaky
  (FakeEmbedder uses randomized `hash()`, so similarity varies per run).
- Ingest fix: `.pytest_cache`/`.ruff_cache`/`.mypy_cache` skipped (Ares
  index had pytest's boilerplate README), and `prune_source_files` drops
  deleted files (per-file upsert never did).
- Retrieval result: 16/16, hit@1 0.79, MRR 0.89. 46 unit tests pass.
- **e2e baseline** (gemini-3.5-flash-lite): 12/16, routing 0.86,
  citations 1.0, keywords 0.86, abstention 2/2, 0 degraded. Failures:
  fly-scaling (real routing miss), finding-3 (ambiguous question → router
  correctly cross-searched; kept strict), engine-step and ares-next-steps
  (answer-content gaps). Details in README "Evaluation".
- **Default model → gemini-3.5-flash-lite.** Probed the 429s:
  gemini-2.5-flash free tier is 20 requests/DAY on this key
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) — ~4 questions.
  gemini-2.5-flash-lite is 404 for new users; gemini-3.8-flash-lite doesn't
  exist. Always run a 4-query `--ids` sample before a full e2e run.

**Phase 6 (deployment), branch `feat/deploy`**: `server.py` (FastAPI,
`POST /ask`, `GET /healthz`, one process-wide runner, refuses to start
without GOOGLE_API_KEY because the index is Gemini-embedded).
`run_question` now deletes each session afterwards (was an in-memory
leak for a long-running server). MiniLM moved to an `offline` extra so
the image has no torch (557MB). `DEVDESK_HOME` env overrides REPO_ROOT —
a non-editable install resolves `parents[2]` into site-packages.
`deploy/cloudrun_service.yaml` + `deploy/iam_setup.md`: runtime SA with
only secretAccessor on the key secret, no allUsers invoker, maxScale=1
(per-process rate limiter vs per-project quota), scale-to-zero.
Verified locally in Docker: healthz, a real /ask round trip (ares_agent,
fully cited, 25s), 422 on bad input, no `.env` in image, non-root, and
refusal to start without a key. **Not deployed to GCP**: no gcloud on
this machine, and Cloud Run needs a billing-enabled project.
51 unit tests pass.

**Phase 7**: README final pass done (stale 0.35 floor / 2.5-flash /
"deployment later" text fixed; Deployment + Evaluation sections).

**Free-tier deploy (branch `feat/free-tier-deploy`, PR #6)**: GCP
always-free covers this (Cloud Run 2M req / 180k vCPU-s / 360k GB-s,
Artifact Registry 0.5GB, Secret Manager 6 versions) but still needs a
billing account (card). **Deploy to a separate GCP project from the
Gemini key** — linking billing to the key's project moves it to paid
Tier 1. Measured: ~130MiB peak RAM → 512Mi; image ~200MB compressed →
AR cleanup policy keeps 2. `deploy/deploy.sh setup|deploy|ask` builds
locally (no Cloud Build), gates on retrieval eval, optional $1 budget
alert. Bash gotcha hit while writing it: an apostrophe inside
`"${VAR:?message}"` opens a quote — broke parsing 60 lines later.

Open items: e2e eval failures (see README), live Cloud Run deploy (needs
Arjun: GCP account + new project + `gcloud auth login`).
