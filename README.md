# DevDesk

A multi-agent RAG assistant built with [Google's Agent Development Kit
(ADK)](https://google.github.io/adk-docs/): a router agent delegates
questions to per-project specialist agents, each grounded in that
project's real documentation through retrieval-augmented generation.

It's built as a demonstration of the pattern an FDE (Applied AI) rebuilds
for every new customer — point the agent at a codebase/doc set (a
"tenant"), and it becomes a support agent for that project. The two
tenants used to build and evaluate it here are the author's own public
projects:

| Tenant | What it is | Source |
|---|---|---|
| `ares` | Local, Ollama-backed multi-agent stack | [Emperor-Z/ares](https://github.com/Emperor-Z/ares) (public, MIT) |
| `verisim` | Simulation-training dissertation project | [Emperor-Z/verisim](https://github.com/Emperor-Z/verisim) (public, MIT) |

New tenants are added by pointing `PROJECTS` in `src/devdesk/config.py` at
a source path and re-running `python -m devdesk.rag.ingest` — nothing else
in the router or the RAG pipeline is tenant-specific.

## Architecture

```mermaid
flowchart TB
    U[User question] --> R[devdesk_router\nADK LlmAgent, ReAct instruction]
    R -->|AgentTool| VA[verisim_agent]
    R -->|AgentTool| AA[ares_agent]
    R -->|tool call| SA[search_all_projects]

    VA --> VS[search_docs project=verisim]
    AA --> AS[search_docs project=ares]
    VA --> VG[git_status_lookup]
    AA --> AG[git_status_lookup]

    VS --> VDB[(Chroma\none collection per project)]
    AS --> VDB
    SA --> VDB

    VDB --> EMB[Embedder\ngemini-embedding-001\nMiniLM offline fallback]

    ING[ingest.py] -->|chunk + embed + upsert| VDB
    DOCS[real project docs\n*.md] --> ING
```

**Delegation pattern**: explicit tool-based delegation, not ADK's automatic
sub-agent transfer. The router's specialists are wrapped as `AgentTool`s,
so routing is a visible, explicit tool call the router reasons about
(ReAct: identify project → call the right tool → compose the answer),
rather than an implicit handoff. This matches how a support agent actually
needs to behave in front of a customer — the decision of *which* system is
being asked about has to be auditable.

**RAG chunking**: markdown-header-aware. A section is kept whole if it fits
inside the token cap; otherwise it's hard-split with overlap. Token counts
always use the MiniLM tokenizer — even when the active embedder is
Gemini — because MiniLM has the stricter 256-token hard cutoff of the two
backends, so sizing every chunk against it keeps both safe from silent
truncation. The chunk cap is 220 tokens with 35-token overlap. Why this
matters for hallucination: a chunk boundary that lands mid-thought hands
the LLM a fragment that reads as ambiguous or contradictory, which is a
common trigger for the model to "fill in" the missing context itself.
Keeping whole subsections together for the common case (structured docs)
avoids that.

**Anti-hallucination behavior**: every specialist is instructed to call
`search_docs` before answering, cite the tool's pre-formatted `citation`
string on every claim, and say plainly when nothing relevant was found
rather than generalizing from training knowledge. `search_docs` itself
enforces a 0.35 cosine-similarity floor — weak matches are dropped rather
than returned, so "no relevant docs" is a real, distinct outcome the agent
can report instead of stretching a bad match into an answer.

## Observability

Two ADK plugins (`google.adk.plugins.base_plugin.BasePlugin`) are wired
into every run via `InMemoryRunner(plugins=[...])` in `cli.py`, applying
uniformly to the router and every specialist without touching agent code:

- **`observability/logging.py` (`StructuredLoggingPlugin`)** — one JSON
  line per step (run/model/tool start, end, error) to stdout and
  `logs/devdesk.jsonl` (gitignored), each with `invocation_id`,
  `agent_name`, and `latency_ms`.
- **`observability/tracing.py` (`LangfuseTracingPlugin`)** — mirrors the
  same steps into a self-hosted Langfuse instance as a trace with
  span/generation children, so real trace viewing comes from existing
  infra rather than a new stack. **No-op** without
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` set — DevDesk works
  identically with or without Langfuse configured.

**Correlating before/after pairs**: ADK constructs a fresh
`CallbackContext` object per callback invocation, so `id(callback_context)`
does *not* identify the same model call across its `before_model_callback`
and `after_model_callback` pair (this looked right in a naive first pass —
every latency read as exactly `0.0` until caught against a live run).
Model calls within one agent's turn run strictly sequentially, so both
plugins key a LIFO stack by `invocation_id` instead.
`ToolContext.function_call_id` is a real per-call identifier and doesn't
have this problem.

## Hardening

Two more plugins, wired in alongside the observability ones:

- **`hardening/rate_limit.py` (`RateLimitPlugin`)** — proactive
  fixed-interval pacing of model calls (`RATE_LIMIT_RPM`, default 8),
  ahead of hitting a 429 rather than reacting to one.
- **`hardening/error_handling.py` (`GracefulDegradationPlugin`)** — turns
  an unhandled model or tool error into a normal, reportable result
  instead of a crash. ADK's `on_tool_error_callback`/
  `on_model_error_callback` can each return a value in place of letting
  the error propagate — that's the mechanism used here, not a new retry
  layer. Verified live by forcing `gemini-3.8-flash`'s 5 req/min free-tier
  limit: the CLI prints a clean "couldn't reach the model right now, try
  again shortly" instead of a stack trace.

Retry itself already exists (`llm_client.py`'s tenacity wrapper for
embeddings, `google-genai`'s own client for chat calls) and isn't
duplicated — `hardening/retry.py` stays an intentionally empty stub
explaining why.

## Setup

Requires Python 3.12 (3.14 has had `chromadb`/dependency compatibility
issues at time of writing — see `context.md`).

```bash
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
# edit .env: set GOOGLE_API_KEY (see below)

python -m devdesk.rag.ingest        # index both tenants
python -m devdesk.cli "what agents make up the Ares stack?"
```

### API keys you need

- **`GOOGLE_API_KEY`** — a free [Google AI Studio](https://aistudio.google.com/apikey)
  key. No GCP project or billing required for this phase; it drives both
  the chat model (`gemini-2.5-flash` by default — see `context.md` for why
  `gemini-3.8-flash` isn't the default yet) and the primary embedding
  model (`gemini-embedding-001`). Without it, embeddings fall back to a
  local offline MiniLM model automatically (chat still needs a key).
- Nothing else is required to run this phase. Deployment (later) will add
  a GCP project + Cloud Run + a switch to `GOOGLE_GENAI_USE_VERTEXAI=TRUE`
  for Vertex AI — not needed yet.

Without a key set, `pytest` still passes in full (all unit tests use a
fake embedder / stub, no network calls); only the one `@pytest.mark.integration`
test is skipped.

- **`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`** — optional. Create a
  "DevDesk" project in a self-hosted Langfuse instance (Settings → API
  Keys) if you want real trace viewing; leave unset and tracing is a
  no-op.

## Status

Phase 2 (agents, tools, RAG), Phase 3 (observability), Phase 4
(hardening) and Phase 5 (eval harness) are built and verified — see
`context.md` for the full build log. Not yet done: deployment.

## Evaluation

```bash
python eval/run_eval.py --mode retrieval   # 1 embedding call/query, no chat model
python eval/run_eval.py --mode e2e         # full router -> specialist loop
python eval/run_eval.py --mode e2e --ids ares-memory none-sourdough --fail-under 1.0
```

16 hand-written queries (`eval/queries.yaml`): 7 VeriSim, 6 Ares, 1
cross-project, 2 out-of-scope. Scoring is deterministic — no LLM judge:

- **retrieval**: hit@k, hit@1, MRR against the expected source files, and
  whether out-of-scope queries are rejected by the similarity cutoff.
- **e2e**: routing accuracy (right specialist, no detour), citation of an
  expected source, required keywords, abstention on out-of-scope queries,
  latency. Answers replaced by the graceful-degradation fallback (free-tier
  503s) are counted as `degraded` and excluded from quality rates.

Latest retrieval run (gemini-embedding-001): hit@k **1.0**, hit@1 **0.79**,
MRR **0.89**, out-of-scope rejected **2/2**. The first run caught a real
bug: the 0.35 similarity cutoff (tuned for MiniLM) let everything through
under Gemini embeddings, whose similarity floor is ~0.5 — a sourdough
question scored 0.51 against the VeriSim docs. Measured relevant hits
bottom out at 0.661 and out-of-scope at 0.621, so the Gemini cutoff is now
0.64, chosen per backend in `config.min_query_similarity()`.

## Repo layout

```
src/devdesk/
  router_agent.py        # top-level LlmAgent, explicit delegation
  specialists/            # one LlmAgent per tenant
  tools/                  # search_docs, search_all_projects, git_status_lookup
  rag/                     # chunking, embeddings, vectorstore, ingest
  cli.py                  # python -m devdesk.cli "question"
  evaluation/             # eval scoring (pure) + runner
eval/queries.yaml          # hand-written eval set
eval/run_eval.py           # python eval/run_eval.py --mode retrieval|e2e
tests/                      # pytest, no network calls in the default run
```
