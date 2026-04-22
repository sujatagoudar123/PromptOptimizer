"""Echo provider.

An offline stub that lets the entire gateway run end-to-end without any
API key. Useful for local dev, tests, CI, and for verifying the pipeline
before plugging in a real provider.

The echo provider returns a deterministic, structured response based on
the prompt so that cache hits are observably identical to fresh calls.
"""
from __future__ import annotations

import asyncio

from ..core.models import ProviderRequest, ProviderResponse
from .base import LLMProvider


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, *, simulate_latency_ms: float = 5.0) -> None:
        self._latency_s = simulate_latency_ms / 1000.0

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self._latency_s > 0:
            await asyncio.sleep(self._latency_s)

        # Deterministic structured echo
        preview = request.prompt.strip().splitlines()[0] if request.prompt.strip() else ""
        if len(preview) > 120:
            preview = preview[:117] + "..."

        completion = (
            f"[echo/{request.model}] Received {len(request.prompt)} characters. "
            f"First line: {preview!r}"
        )

        # Fake token count: 1 token per 4 chars, min 1
        completion_tokens = max(1, len(completion) // 4)

        return ProviderResponse(
            completion=completion,
            provider=self.name,
            model=request.model,
            completion_tokens=completion_tokens,
            raw={"echo": True},
        )
