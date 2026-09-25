# Suggested commands

Working directory: /home/z/devdesk

- Install (editable, with dev deps): `pip install -e ".[dev]"`
- Run tests: `pytest`
- Lint: `ruff check .`
- Type check: `mypy src/`
- Run eval harness (once built): `python eval/run_eval.py`
- Local Chroma + agent via compose (once Dockerfile/compose filled in):
  `docker compose up`
- Git: standard git; no Claude attribution lines in commits (owner's
  standing rule — see global memory `feedback_no_claude_attribution`)

No CI/lint config committed yet — add `.ruff.toml`/`pyproject` tool sections
when the first real modules land instead of leaving default settings.
