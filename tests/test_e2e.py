"""End-to-end tests — exercise the whole pipeline through Gateway.complete."""
from __future__ import annotations

import pytest

from llm_gateway.core.exceptions import EmptyPromptError
from llm_gateway.core.models import CacheStatus, CompletionRequest, RoutingPath


@pytest.mark.asyncio
async def test_happy_path(gateway):
    req = CompletionRequest(prompt="Could you please summarize the report?")
    resp = await gateway.complete(req)
    assert resp.completion.startswith("[echo/")
    assert resp.metadata.cache_status == CacheStatus.MISS
    assert resp.metadata.routing_path == RoutingPath.UPSTREAM
    assert resp.metadata.provider == "echo"
    # Optimization should have at least normalized or compressed
    assert resp.metadata.optimization.applied is True


@pytest.mark.asyncio
async def test_exact_cache_hit_on_second_call(gateway):
    req = CompletionRequest(prompt="What is the capital of France?")
    r1 = await gateway.complete(req)
    r2 = await gateway.complete(req)
    assert r1.metadata.cache_status == CacheStatus.MISS
    assert r2.metadata.cache_status == CacheStatus.EXACT_HIT
    assert r2.metadata.routing_path == RoutingPath.CACHE
    assert r2.completion == r1.completion


@pytest.mark.asyncio
async def test_semantic_cache_hit(gateway):
    r1 = await gateway.complete(
        CompletionRequest(prompt="Please summarize the quarterly revenue report")
    )
    assert r1.metadata.cache_status == CacheStatus.MISS

    # Near-identical prompt with different filler — should semantically match
    r2 = await gateway.complete(
        CompletionRequest(prompt="Could you please summarize the quarterly revenue report.")
    )
    assert r2.metadata.cache_status in (
        CacheStatus.EXACT_HIT,
        CacheStatus.SEMANTIC_HIT,
    )


@pytest.mark.asyncio
async def test_bypass_cache_forces_upstream(gateway):
    req = CompletionRequest(prompt="stable prompt for bypass test")
    await gateway.complete(req)  # warm
    req2 = CompletionRequest(prompt="stable prompt for bypass test", bypass_cache=True)
    r = await gateway.complete(req2)
    assert r.metadata.cache_status == CacheStatus.BYPASS
    assert r.metadata.routing_path == RoutingPath.UPSTREAM


@pytest.mark.asyncio
async def test_bypass_optimization_preserves_prompt(gateway):
    original = "Please kindly do X. Please kindly do X."
    # Bypass BOTH layers so the sent prompt is preserved verbatim.
    req = CompletionRequest(
        prompt=original,
        bypass_optimization=True,
        bypass_coaching=True,
    )
    r = await gateway.complete(req)
    assert r.metadata.optimized_prompt == original
    assert r.metadata.optimization.applied is False


@pytest.mark.asyncio
async def test_empty_prompt_raises(gateway):
    with pytest.raises(EmptyPromptError):
        await gateway.complete(CompletionRequest(prompt="   "))


@pytest.mark.asyncio
async def test_tokens_saved_reported(gateway):
    req = CompletionRequest(
        prompt=(
            "Could you please kindly summarize this report. "
            "I would like you to do that as soon as possible. "
            "Please kindly be very concise."
        )
    )
    r = await gateway.complete(req)
    assert r.metadata.tokens.tokens_saved >= 0
    assert r.metadata.tokens.original_prompt_tokens >= r.metadata.tokens.optimized_prompt_tokens


@pytest.mark.asyncio
async def test_unsafe_to_cache_never_caches(gateway):
    req = CompletionRequest(
        prompt="user-specific secret prompt",
        unsafe_to_cache=True,
    )
    r1 = await gateway.complete(req)
    r2 = await gateway.complete(
        CompletionRequest(prompt="user-specific secret prompt")
    )
    # Second call must still be a miss since the first wasn't stored
    assert r1.metadata.cache_status == CacheStatus.MISS
    assert r2.metadata.cache_status == CacheStatus.MISS


@pytest.mark.asyncio
async def test_failover_when_primary_fails(metrics):
    """Provider chain failover: the first provider raises, the second answers."""
    from llm_gateway.caching.manager import CacheManager
    from llm_gateway.caching.memory import MemoryCache
    from llm_gateway.core.exceptions import ProviderError
    from llm_gateway.core.models import ProviderRequest, ProviderResponse
    from llm_gateway.core.orchestrator import Gateway
    from llm_gateway.governance.policy import GovernancePolicy
    from llm_gateway.governance.tokens import build_estimator
    from llm_gateway.optimization.optimizer import PromptOptimizer
    from llm_gateway.providers.base import LLMProvider
    from llm_gateway.providers.echo import EchoProvider
    from llm_gateway.routing.router import Router

    class AlwaysFail(LLMProvider):
        name = "broken"
        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            raise ProviderError("simulated failure")

    estimator = build_estimator()
    from llm_gateway.coaching.coach import PromptCoach
    from llm_gateway.coaching.scorer import PromptQualityScorer
    gw = Gateway(
        scorer=PromptQualityScorer(),
        coach=PromptCoach(),
        optimizer=PromptOptimizer(),
        cache=CacheManager(
            backend=MemoryCache(),
            embedder=None,
            semantic_enabled=False,
            similarity_threshold=0.9,
            metrics=metrics,
        ),
        router=Router([AlwaysFail(), EchoProvider()], metrics=metrics),
        governance=GovernancePolicy(estimator, max_prompt_tokens=10000, reject_oversize=False),
        token_estimator=estimator,
        metrics=metrics,
        default_model="echo-1",
        coaching_enabled=True,
        prompt_quality_threshold=0.55,
    )
    r = await gw.complete(CompletionRequest(prompt="failover test"))
    assert r.metadata.routing_path == RoutingPath.FAILOVER
    assert r.metadata.provider == "echo"
    await gw.aclose()


@pytest.mark.asyncio
async def test_all_providers_failed(metrics):
    from llm_gateway.caching.manager import CacheManager
    from llm_gateway.caching.memory import MemoryCache
    from llm_gateway.core.exceptions import AllProvidersFailedError, ProviderError
    from llm_gateway.core.models import ProviderRequest, ProviderResponse
    from llm_gateway.core.orchestrator import Gateway
    from llm_gateway.governance.policy import GovernancePolicy
    from llm_gateway.governance.tokens import build_estimator
    from llm_gateway.optimization.optimizer import PromptOptimizer
    from llm_gateway.providers.base import LLMProvider
    from llm_gateway.routing.router import Router

    class Broken(LLMProvider):
        name = "broken"
        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            raise ProviderError("down")

    estimator = build_estimator()
    from llm_gateway.coaching.coach import PromptCoach
    from llm_gateway.coaching.scorer import PromptQualityScorer
    gw = Gateway(
        scorer=PromptQualityScorer(),
        coach=PromptCoach(),
        optimizer=PromptOptimizer(),
        cache=CacheManager(
            backend=MemoryCache(),
            embedder=None,
            semantic_enabled=False,
            similarity_threshold=0.9,
            metrics=metrics,
        ),
        router=Router([Broken(), Broken()], metrics=metrics),
        governance=GovernancePolicy(estimator, max_prompt_tokens=10000, reject_oversize=False),
        token_estimator=estimator,
        metrics=metrics,
        default_model="x",
        coaching_enabled=True,
        prompt_quality_threshold=0.55,
    )
    with pytest.raises(AllProvidersFailedError):
        await gw.complete(CompletionRequest(prompt="no providers will answer"))
    await gw.aclose()
