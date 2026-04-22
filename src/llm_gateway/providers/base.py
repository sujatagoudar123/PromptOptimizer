"""LLM provider adapter interface.

All upstream providers implement this contract. The routing layer calls
``complete`` asynchronously and expects either a ProviderResponse or
a ProviderError / ProviderTimeoutError.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.models import ProviderRequest, ProviderResponse


class LLMProvider(ABC):
    """Abstract upstream LLM provider."""

    name: str = "abstract"

    @abstractmethod
    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Perform a completion. Raise ProviderError on failure."""

    async def aclose(self) -> None:
        """Release any resources (HTTP clients, etc). Default: no-op."""
        return None
