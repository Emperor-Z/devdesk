"""Proactive rate limiting for model calls, as an ADK plugin.

Reactive retry (llm_client.py, ADK's own google-genai client) handles a
429 after it happens. This paces calls *before* they'd hit one — the
dominant real bottleneck observed in practice (a single multi-agent-hop
question can burn 5+ model calls, and gemini-3.8-flash's free tier is
only 5 req/min).
"""

from __future__ import annotations

import asyncio
import os
import time

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin

DEFAULT_RPM = 8


class RateLimitPlugin(BasePlugin):
    """Fixed-interval pacing (a size-1 leaky bucket): spaces
    before_model_callback calls at least 60/rpm seconds apart, blocking
    (awaiting) rather than rejecting. Simple on purpose — this isn't
    trying to allow bursts, just to stay under a free-tier RPM ceiling."""

    def __init__(
        self,
        name: str = "rate_limit",
        rpm: int | None = None,
        clock=time.monotonic,
        sleep=asyncio.sleep,
    ) -> None:
        super().__init__(name)
        self._rpm = rpm or int(os.environ.get("RATE_LIMIT_RPM", DEFAULT_RPM))
        self._interval = 60.0 / self._rpm
        self._clock = clock
        self._sleep = sleep
        self._next_slot: float | None = None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        now = self._clock()
        if self._next_slot is None or now >= self._next_slot:
            self._next_slot = now + self._interval
            return
        wait = self._next_slot - now
        self._next_slot += self._interval
        await self._sleep(wait)
        return
