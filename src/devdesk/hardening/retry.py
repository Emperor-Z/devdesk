"""Intentionally empty.

Retry already exists and isn't duplicated here: `llm_client.py`'s
tenacity-based `with_retry` covers our own embedding calls, and ADK's
`google-genai` client retries chat calls internally with backoff before
raising. A third independent retry layer on top would just multiply
backoff delays. See `rate_limit.py` (proactive pacing) and
`error_handling.py` (graceful fallback once retries are exhausted) for
what this phase actually adds.
"""
