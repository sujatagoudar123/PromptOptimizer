"""Layered cache manager.

Fronts two physical stores:
  * exact:    lookup by SHA-256 of the optimized prompt + model
  * semantic: lookup by embedding cosine similarity above a threshold

The manager isolates cache backend errors: a failing backend degrades
the request to a cache miss rather than failing the whole gateway.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..core.exceptions import CacheBackendError
from ..core.models import CacheStatus
from ..observability.logging import get_logger
from ..observability.metrics import Metrics
from .base import CacheBackend, CacheEntry
from .embeddings import Embedder, cosine_similarity


log = get_logger(__name__)


@dataclass
class CacheLookupResult:
    status: CacheStatus
    value: str | None
    similarity: float | None = None  # only for semantic hits


def _exact_key(prompt: str, model: str) -> str:
    """Stable hash over (model, prompt) for exact lookups."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return f"exact:{h.hexdigest()}"


def _semantic_key(prompt: str, model: str) -> str:
    """Semantic entries are also keyed uniquely, but retrieval is via scan."""
    h = hashlib.sha256()
    h.update(b"sem:")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return f"semantic:{h.hexdigest()}"


class CacheManager:
    def __init__(
        self,
        *,
        backend: CacheBackend,
        embedder: Embedder | None,
        semantic_enabled: bool,
        similarity_threshold: float,
        metrics: Metrics,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._semantic_enabled = semantic_enabled and embedder is not None
        self._threshold = similarity_threshold
        self._metrics = metrics

    async def lookup(self, prompt: str, model: str) -> CacheLookupResult:
        # 1. Exact lookup
        try:
            entry = await self._backend.get(_exact_key(prompt, model))
        except CacheBackendError as e:
            log.warning("cache.exact_lookup_failed", error=str(e))
            self._metrics.record_cache_miss()
            return CacheLookupResult(status=CacheStatus.MISS, value=None)

        if entry is not None:
            self._metrics.record_cache_hit("exact")
            return CacheLookupResult(status=CacheStatus.EXACT_HIT, value=entry.value)

        # 2. Semantic lookup
        if self._semantic_enabled and self._embedder is not None:
            try:
                query_vec = await self._embedder.embed(prompt)
                best_entry: CacheEntry | None = None
                best_sim = 0.0
                async for cached in self._backend.iter_entries():
                    if cached.embedding is None:
                        continue
                    # Semantic entries are namespaced; exact entries are skipped.
                    if not cached.key.startswith("semantic:"):
                        continue
                    sim = cosine_similarity(query_vec, cached.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = cached

                if best_entry is not None and best_sim >= self._threshold:
                    self._metrics.record_cache_hit("semantic")
                    return CacheLookupResult(
                        status=CacheStatus.SEMANTIC_HIT,
                        value=best_entry.value,
                        similarity=best_sim,
                    )
            except CacheBackendError as e:
                log.warning("cache.semantic_lookup_failed", error=str(e))
            except Exception as e:  # embedding service failures
                log.warning("cache.embedding_failed", error=str(e))

        self._metrics.record_cache_miss()
        return CacheLookupResult(status=CacheStatus.MISS, value=None)

    async def store(
        self, prompt: str, model: str, value: str, *, unsafe_to_cache: bool = False
    ) -> None:
        if unsafe_to_cache:
            return
        # Exact entry
        try:
            await self._backend.set(
                CacheEntry(key=_exact_key(prompt, model), value=value)
            )
        except CacheBackendError as e:
            log.warning("cache.exact_store_failed", error=str(e))
            return

        # Semantic entry
        if self._semantic_enabled and self._embedder is not None:
            try:
                vec = await self._embedder.embed(prompt)
                await self._backend.set(
                    CacheEntry(
                        key=_semantic_key(prompt, model),
                        value=value,
                        embedding=vec,
                    )
                )
            except CacheBackendError as e:
                log.warning("cache.semantic_store_failed", error=str(e))
            except Exception as e:
                log.warning("cache.semantic_embed_failed", error=str(e))

    async def size(self) -> int:
        try:
            return await self._backend.size()
        except CacheBackendError:
            return 0

    async def clear(self) -> None:
        try:
            await self._backend.clear()
        except CacheBackendError as e:
            log.warning("cache.clear_failed", error=str(e))
