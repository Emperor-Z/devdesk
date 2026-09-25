"""Shared retry policy for Gemini API calls (chat + embeddings).

Free-tier keys hit 429s under normal use (each question costs at least two
LLM calls: router, then specialist). Pulled forward from the hardening
phase because without it, smoke tests and eval runs flake at random.

Also forces IPv4 for outbound connections: some environments (this one
included) have a broken/blackholed IPv6 route, which makes every gRPC/HTTP
connection attempt burn 60-130s on a dead SYN handshake before falling
back to IPv4. That's not a rate limit or a code bug, just a slow default —
worth ruling out first if a call looks hung rather than failed.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(*args, **kwargs):
    results = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in results if r[0] == socket.AF_INET]
    return ipv4 or results


socket.getaddrinfo = _ipv4_only_getaddrinfo


def _is_rate_limit_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, "429", "RESOURCE_EXHAUSTED"):
        return True
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


gemini_retry = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


def with_retry(fn: Callable[[], T]) -> T:
    return gemini_retry(fn)()
