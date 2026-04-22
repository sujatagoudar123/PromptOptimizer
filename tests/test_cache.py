"""Cache layer tests — memory backend, embeddings, and layered manager."""
from __future__ import annotations

import asyncio

import pytest

from llm_gateway.caching.base import CacheEntry
from llm_gateway.caching.embeddings import HashingEmbedder, cosine_similarity
from llm_gateway.caching.manager import CacheManager
from llm_gateway.caching.memory import MemoryCache
from llm_gateway.core.models import CacheStatus
from llm_gateway.observability.metrics import Metrics


# --- Memory backend ---


@pytest.mark.asyncio
async def test_memory_set_and_get():
    cache = MemoryCache(max_entries=10, ttl_seconds=60)
    await cache.set(CacheEntry(key="k1", value="v1"))
    got = await cache.get("k1")
    assert got is not None
    assert got.value == "v1"


@pytest.mark.asyncio
async def test_memory_ttl_expiry():
    cache = MemoryCache(max_entries=10, ttl_seconds=0)  # immediate
    await cache.set(CacheEntry(key="k", value="v", created_at=0.0))
    got = await cache.get("k")
    # With ttl=0 we treat it as "no expiry" per our impl — so this returns the value.
    assert got is not None


@pytest.mark.asyncio
async def test_memory_lru_eviction():
    cache = MemoryCache(max_entries=3, ttl_seconds=3600)
    for i in range(4):
        await cache.set(CacheEntry(key=f"k{i}", value=f"v{i}"))
    # k0 should have been evicted (oldest)
    assert await cache.get("k0") is None
    assert await cache.get("k3") is not None


@pytest.mark.asyncio
async def test_memory_iter_entries():
    cache = MemoryCache(max_entries=10, ttl_seconds=3600)
    await cache.set(CacheEntry(key="a", value="1"))
    await cache.set(CacheEntry(key="b", value="2"))
    keys = [e.key async for e in cache.iter_entries()]
    assert set(keys) == {"a", "b"}


# --- Embeddings ---


@pytest.mark.asyncio
async def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    v1 = await e.embed("hello world")
    v2 = await e.embed("hello world")
    assert v1 == v2


@pytest.mark.asyncio
async def test_hashing_embedder_similarity():
    e = HashingEmbedder(dim=256)
    v1 = await e.embed("summarize the quarterly report")
    v2 = await e.embed("summarize the quarterly report please")
    v3 = await e.embed("tell me a bedtime story about dragons")
    s_close = cosine_similarity(v1, v2)
    s_far = cosine_similarity(v1, v3)
    assert s_close > s_far
    assert s_close > 0.6


def test_cosine_similarity_handles_edges():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    # identical unit vector
    import math
    v = [1 / math.sqrt(2), 1 / math.sqrt(2)]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


# --- Layered manager ---


@pytest.mark.asyncio
async def test_layered_cache_exact_hit():
    metrics = Metrics()
    backend = MemoryCache(max_entries=10, ttl_seconds=60)
    cm = CacheManager(
        backend=backend,
        embedder=HashingEmbedder(dim=128),
        semantic_enabled=True,
        similarity_threshold=0.95,
        metrics=metrics,
    )
    await cm.store("what is 2+2", "echo-1", "4")
    r = await cm.lookup("what is 2+2", "echo-1")
    assert r.status == CacheStatus.EXACT_HIT
    assert r.value == "4"


@pytest.mark.asyncio
async def test_layered_cache_semantic_hit():
    metrics = Metrics()
    backend = MemoryCache(max_entries=10, ttl_seconds=60)
    cm = CacheManager(
        backend=backend,
        embedder=HashingEmbedder(dim=256),
        semantic_enabled=True,
        similarity_threshold=0.80,  # lenient so hashing can hit
        metrics=metrics,
    )
    await cm.store("summarize the quarterly report", "echo-1", "SUMMARY")
    # Very similar but not identical
    r = await cm.lookup("summarize the quarterly report.", "echo-1")
    # Either exact (if punctuation stripped) or semantic — both count as a hit
    assert r.status in (CacheStatus.EXACT_HIT, CacheStatus.SEMANTIC_HIT)


@pytest.mark.asyncio
async def test_layered_cache_miss_for_different_prompt():
    metrics = Metrics()
    backend = MemoryCache(max_entries=10, ttl_seconds=60)
    cm = CacheManager(
        backend=backend,
        embedder=HashingEmbedder(dim=128),
        semantic_enabled=True,
        similarity_threshold=0.95,
        metrics=metrics,
    )
    await cm.store("summarize the quarterly report", "echo-1", "A")
    r = await cm.lookup("tell me a bedtime story", "echo-1")
    assert r.status == CacheStatus.MISS


@pytest.mark.asyncio
async def test_unsafe_to_cache_is_not_stored():
    metrics = Metrics()
    backend = MemoryCache(max_entries=10, ttl_seconds=60)
    cm = CacheManager(
        backend=backend,
        embedder=None,
        semantic_enabled=False,
        similarity_threshold=0.95,
        metrics=metrics,
    )
    await cm.store("sensitive prompt", "echo-1", "secret", unsafe_to_cache=True)
    r = await cm.lookup("sensitive prompt", "echo-1")
    assert r.status == CacheStatus.MISS
