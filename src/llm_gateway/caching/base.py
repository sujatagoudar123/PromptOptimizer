"""Cache backend abstraction.

Two concrete backends ship with the gateway:
  * MemoryCache — in-process LRU with TTL (zero ops; default)
  * RedisCache  — distributed, shared across gateway replicas (optional)

Both implement the same async interface. Semantic cache requires an
additional list-scan capability exposed by ``iter_entries``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class CacheEntry:
    key: str
    value: str
    # Embedding is only set for the semantic layer. None for exact-hash entries.
    embedding: list[float] | None = None
    created_at: float = 0.0  # epoch seconds


class CacheBackend(ABC):
    """Abstract async cache backend."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None: ...

    @abstractmethod
    async def set(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def size(self) -> int: ...

    @abstractmethod
    def iter_entries(self) -> AsyncIterator[CacheEntry]:
        """Scan all live (non-expired) entries. Used by the semantic layer."""
        ...
