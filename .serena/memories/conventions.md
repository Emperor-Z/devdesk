# Code style & conventions

- Full type hints on all function signatures.
- Docstrings only where they explain non-obvious WHY (design constraint,
  workaround, invariant) — not restating what the code does. Matches the
  owner's general no-filler-comment preference.
- No premature abstraction: this is a small, readable codebase intended to
  read well in a code review (portfolio piece) — prefer explicit, flat code
  over frameworks-within-the-framework.
- Free-tier-only constraint is load-bearing: no paid embedding/LLM APIs, no
  paid vector DB, no paid deploy tier. Any dependency choice must be
  justified against this before adding it.
- Tests live under tests/, one file per major area (agents, tools, rag).
- No Claude/AI attribution added to commits or files (global rule).
