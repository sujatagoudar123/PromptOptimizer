"""OpenAI provider adapter.

Uses the Chat Completions REST API directly via httpx so we don't depend
on the OpenAI SDK's opinions about retries, logging, etc.
"""
from __future__ import annotations

import httpx

from ..core.exceptions import ProviderError, ProviderTimeoutError
from ..core.models import ProviderRequest, ProviderResponse
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "OpenAI provider requires an API key.",
                detail="Set LLMGW_OPENAI_API_KEY.",
            )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                "OpenAI request timed out.", detail=str(e)
            ) from e
        except httpx.HTTPError as e:
            raise ProviderError("OpenAI HTTP error.", detail=str(e)) from e

        if resp.status_code >= 500:
            raise ProviderError(
                f"OpenAI returned {resp.status_code}.",
                detail=resp.text[:500],
            )
        if resp.status_code >= 400:
            # 4xx is surfaced but treated as provider error so the router
            # can choose to failover to the next provider if desired.
            raise ProviderError(
                f"OpenAI rejected request (HTTP {resp.status_code}).",
                detail=resp.text[:500],
            )

        data = resp.json()
        try:
            completion = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(
                "Unexpected OpenAI response shape.", detail=str(data)[:500]
            ) from e

        usage = data.get("usage", {}) or {}

        return ProviderResponse(
            completion=completion,
            provider=self.name,
            model=data.get("model", request.model),
            completion_tokens=usage.get("completion_tokens"),
            raw={"usage": usage, "id": data.get("id")},
        )

    async def aclose(self) -> None:
        await self._client.aclose()
