"""Routing layer.

Responsibilities:
  * decide whether a request can be served from cache (handled by
    the orchestrator — the router is invoked only on cache miss)
  * call the primary provider
  * on provider failure/timeout, failover to the next provider in the chain
  * emit metrics and structured logs for each attempt
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..core.exceptions import (
    AllProvidersFailedError,
    ProviderError,
    ProviderTimeoutError,
)
from ..core.models import ProviderRequest, ProviderResponse, RoutingPath
from ..observability.logging import get_logger
from ..observability.metrics import Metrics
from ..providers.base import LLMProvider


log = get_logger(__name__)


@dataclass
class RouteResult:
    response: ProviderResponse
    path: RoutingPath
    attempts: int
    upstream_latency_ms: float


class Router:
    def __init__(self, providers: list[LLMProvider], *, metrics: Metrics) -> None:
        if not providers:
            raise ValueError("Router requires at least one provider.")
        self._providers = providers
        self._metrics = metrics

    async def complete(self, request: ProviderRequest) -> RouteResult:
        attempts = 0
        last_error: Exception | None = None

        for idx, provider in enumerate(self._providers):
            attempts += 1
            path = RoutingPath.UPSTREAM if idx == 0 else RoutingPath.FAILOVER

            started = time.perf_counter()
            try:
                response = await provider.complete(request)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._metrics.upstream_latency_seconds.labels(
                    provider=provider.name
                ).observe(elapsed_ms / 1000.0)
                log.info(
                    "router.success",
                    provider=provider.name,
                    attempt=attempts,
                    latency_ms=round(elapsed_ms, 2),
                )
                return RouteResult(
                    response=response,
                    path=path,
                    attempts=attempts,
                    upstream_latency_ms=elapsed_ms,
                )
            except (ProviderError, ProviderTimeoutError) as e:
                self._metrics.record_provider_failure(provider.name)
                log.warning(
                    "router.provider_failed",
                    provider=provider.name,
                    attempt=attempts,
                    error=str(e),
                )
                last_error = e
                continue

        raise AllProvidersFailedError(
            "All configured providers failed.",
            detail=str(last_error) if last_error else None,
        )

    async def aclose(self) -> None:
        for p in self._providers:
            try:
                await p.aclose()
            except Exception:  # pragma: no cover
                pass
