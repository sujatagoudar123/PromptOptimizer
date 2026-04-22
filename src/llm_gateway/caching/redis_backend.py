"""Redis-backed cache (optional).

Only instantiated when LLMGW_CACHE_BACKEND=redis. Uses redis.asyncio
from the ``redis`` package (installed on demand). Entries are stored
as JSON under a configurable key prefix.

For single-node / local development, prefer MemoryCache.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from ..core.exceptions import CacheBackendError
from .base import CacheBackend, CacheEntry


class RedisCache(CacheBackend):
    """Redis implementation of CacheBackend."""

    def __init__(
        self,
        url: str,
        *,
        ttl_seconds: int = 3600,
        key_prefix: str = "llmgw:",
    ) -> None:
        try:
            import redis.asyncio as redis  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise CacheBackendError(
                "Redis backend requested but 'redis' package is not installed. "
                "Run: pip install redis"
            ) from e

        self._redis = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _serialize(self, entry: CacheEntry) -> str:
        return json.dumps(
            {
                "key": entry.key,
                "value": entry.value,
                "embedding": entry.embedding,
                "created_at": entry.created_at,
            }
        )

    def _deserialize(self, raw: str) -> CacheEntry:
        data = json.loads(raw)
        return CacheEntry(
            key=data["key"],
            value=data["value"],
            embedding=data.get("embedding"),
            created_at=float(data.get("created_at", 0.0)),
        )

    async def get(self, key: str) -> CacheEntry | None:
        try:
            raw = await self._redis.get(self._k(key))
        except Exception as e:  # pragma: no cover - IO dependent
            raise CacheBackendError("Redis GET failed", detail=str(e)) from e
        if raw is None:
            return None
        return self._deserialize(raw)

    async def set(self, entry: CacheEntry) -> None:
        if not entry.created_at:
            entry.created_at = time.time()
        try:
            await self._redis.set(
                self._k(entry.key),
                self._serialize(entry),
                ex=self._ttl if self._ttl > 0 else None,
            )
        except Exception as e:  # pragma: no cover
            raise CacheBackendError("Redis SET failed", detail=str(e)) from e

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(self._k(key))
        except Exception as e:  # pragma: no cover
            raise CacheBackendError("Redis DELETE failed", detail=str(e)) from e

    async def clear(self) -> None:
        # Only clears keys under our prefix; never FLUSHDB.
        try:
            async for key in self._redis.scan_iter(match=f"{self._prefix}*"):
                await self._redis.delete(key)
        except Exception as e:  # pragma: no cover
            raise CacheBackendError("Redis CLEAR failed", detail=str(e)) from e

    async def size(self) -> int:
        count = 0
        try:
            async for _ in self._redis.scan_iter(match=f"{self._prefix}*"):
                count += 1
        except Exception as e:  # pragma: no cover
            raise CacheBackendError("Redis SIZE failed", detail=str(e)) from e
        return count

    async def iter_entries(self) -> AsyncIterator[CacheEntry]:
        try:
            async for k in self._redis.scan_iter(match=f"{self._prefix}*"):
                raw = await self._redis.get(k)
                if raw:
                    yield self._deserialize(raw)
        except Exception as e:  # pragma: no cover
            raise CacheBackendError("Redis SCAN failed", detail=str(e)) from e
