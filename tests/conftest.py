"""Shared pytest fixtures.

All tests run fully offline against the EchoProvider and HashingEmbedder.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio

# Set env before any settings import so get_settings() picks them up.
os.environ.setdefault("LLMGW_ENVIRONMENT", "development")
os.environ.setdefault("LLMGW_PROVIDER_CHAIN", "echo")
os.environ.setdefault("LLMGW_CACHE_BACKEND", "memory")
os.environ.setdefault("LLMGW_EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("LLMGW_LOG_LEVEL", "WARNING")

from llm_gateway.caching.embeddings import HashingEmbedder
from llm_gateway.caching.manager import CacheManager
from llm_gateway.caching.memory import MemoryCache
from llm_gateway.coaching.coach import PromptCoach
from llm_gateway.coaching.scorer import PromptQualityScorer
from llm_gateway.config.settings import Settings, get_settings
from llm_gateway.core.orchestrator import Gateway
from llm_gateway.governance.policy import GovernancePolicy
from llm_gateway.governance.tokens import build_estimator
from llm_gateway.observability.logging import configure_logging
from llm_gateway.observability.metrics import Metrics
from llm_gateway.optimization.optimizer import PromptOptimizer
from llm_gateway.providers.echo import EchoProvider
from llm_gateway.routing.router import Router


configure_logging("WARNING")


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


@pytest_asyncio.fixture
async def gateway(settings, metrics) -> Gateway:
    estimator = build_estimator()
    governance = GovernancePolicy(
        estimator,
        max_prompt_tokens=settings.max_prompt_tokens,
        reject_oversize=settings.reject_oversize,
    )
    optimizer = PromptOptimizer(enabled=True)
    backend = MemoryCache(max_entries=100, ttl_seconds=60)
    cache = CacheManager(
        backend=backend,
        embedder=HashingEmbedder(dim=256),
        semantic_enabled=True,
        # Hashing embedder produces lower similarity scores than a real
        # embedding model, so we use a threshold tuned for it in tests.
        # Production defaults (0.92) assume a proper embedding model.
        similarity_threshold=0.80,
        metrics=metrics,
    )
    router = Router([EchoProvider(simulate_latency_ms=0.0)], metrics=metrics)
    gw = Gateway(
        scorer=PromptQualityScorer(),
        coach=PromptCoach(),
        optimizer=optimizer,
        cache=cache,
        router=router,
        governance=governance,
        token_estimator=estimator,
        metrics=metrics,
        default_model="echo-1",
        coaching_enabled=True,
        prompt_quality_threshold=0.55,
    )
    yield gw
    await gw.aclose()
