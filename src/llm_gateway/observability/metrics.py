"""Prometheus metrics for the gateway.

Exposed via /metrics. All metrics use a dedicated registry so tests
can reset state without polluting the global default registry.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge


class Metrics:
    """Bundle of all gateway counters, gauges, and histograms.

    Using a dedicated registry per Metrics() instance keeps tests isolated.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.requests_total = Counter(
            "llmgw_requests_total",
            "Total number of completion requests received.",
            registry=self.registry,
        )
        self.errors_total = Counter(
            "llmgw_errors_total",
            "Total errors by code.",
            labelnames=("code",),
            registry=self.registry,
        )
        self.cache_hits_total = Counter(
            "llmgw_cache_hits_total",
            "Cache hits by kind.",
            labelnames=("kind",),  # exact | semantic
            registry=self.registry,
        )
        self.cache_misses_total = Counter(
            "llmgw_cache_misses_total",
            "Cache misses.",
            registry=self.registry,
        )
        self.tokens_saved_total = Counter(
            "llmgw_tokens_saved_total",
            "Total prompt tokens saved by optimization.",
            registry=self.registry,
        )
        self.provider_failures_total = Counter(
            "llmgw_provider_failures_total",
            "Upstream provider failures.",
            labelnames=("provider",),
            registry=self.registry,
        )
        self.optimization_bypasses_total = Counter(
            "llmgw_optimization_bypasses_total",
            "Requests that bypassed optimization.",
            registry=self.registry,
        )
        self.coaching_applied_total = Counter(
            "llmgw_coaching_applied_total",
            "Requests where the Coach rewrote the prompt due to low quality.",
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "llmgw_request_latency_seconds",
            "End-to-end gateway latency.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )
        self.upstream_latency_seconds = Histogram(
            "llmgw_upstream_latency_seconds",
            "Latency of upstream provider calls.",
            labelnames=("provider",),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.cache_entries = Gauge(
            "llmgw_cache_entries",
            "Current number of entries in the cache.",
            labelnames=("kind",),
            registry=self.registry,
        )

    # --- Convenience helpers ---

    def record_cache_hit(self, kind: str) -> None:
        self.cache_hits_total.labels(kind=kind).inc()

    def record_cache_miss(self) -> None:
        self.cache_misses_total.inc()

    def record_provider_failure(self, provider: str) -> None:
        self.provider_failures_total.labels(provider=provider).inc()

    def record_error(self, code: str) -> None:
        self.errors_total.labels(code=code).inc()
