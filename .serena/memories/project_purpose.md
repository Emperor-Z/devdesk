# DevDesk — Purpose

Multi-agent RAG assistant built with Google's Agent Development Kit (ADK).
Portfolio piece for a Google Forward Deployed Engineer (Applied AI)
application, framed externally as a generalizable multi-tenant
customer-onboarding pattern: point it at any codebase/doc set and it becomes
a support agent for that project.

The two tenants used to build/eval it are the owner's own public repos:
- VeriSim / BayTrainer (dissertation project)
- Ares (local agent stack)

Full narrative/decision history and build-order checklist: see
`/home/z/devdesk/context.md` at the repo root — read that first when
resuming work, it is the source of truth for project status.

Architecture: router agent (ReAct loop) delegates to per-project specialist
agents, each with its own Chroma vector store. Tools: search_docs (scoped),
search_all_projects (cross-tenant), git_status_lookup. `doc_generator` is
an empty, deliberately deferred stub — nothing in the index is
LLM-synthesized.
