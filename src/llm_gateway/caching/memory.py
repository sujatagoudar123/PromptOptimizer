"""In-process async LRU+TTL cache.

Thread-safe via asyncio.Lock. Suitable for single-replica deployments
and local development. For multi-replica, switch to RedisCache.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import AsyncIterator

from .base import CacheBackend, CacheEntry


class MemoryCache(CacheBackend):
    def __init__(self, *, max_entries: int = 10_000, ttl_seconds: int = 3600) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    def _is_expired(self, entry: CacheEntry) -> bool:
        if self._ttl <= 0:
            return False
        return (time.time() - entry.created_at) > self._ttl

    async def get(self, key: str) -> CacheEntry | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                return None
            # LRU bump
            self._store.move_to_end(key)
            return entry

    async def set(self, entry: CacheEntry) -> None:
        if not entry.created_at:
            entry.created_at = time.time()
        async with self._lock:
            if entry.key in self._store:
                self._store.move_to_end(entry.key)
            self._store[entry.key] = entry
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)

    async def iter_entries(self) -> AsyncIterator[CacheEntry]:
        # Snapshot under lock, then yield outside it so iteration
        # doesn't hold the lock for the duration of a full scan.
        async with self._lock:
            snapshot = list(self._store.values())
        for entry in snapshot:
            if not self._is_expired(entry):
                yield entry
