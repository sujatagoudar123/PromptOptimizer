"""Provider registry / factory.

Builds the ordered list of providers based on the configured chain.
The first provider in the chain is primary; the rest are failover
targets in order.
"""
from __future__ import annotations

from ..config.settings import Settings
from ..core.exceptions import ConfigurationError
from .base import LLMProvider
from .echo import EchoProvider
from .openai_provider import OpenAIProvider


def build_providers(settings: Settings) -> list[LLMProvider]:
    chain = settings.provider_list
    if not chain:
        raise ConfigurationError("LLMGW_PROVIDER_CHAIN must not be empty.")

    providers: list[LLMProvider] = []
    for name in chain:
        name = name.lower()
        if name == "echo":
            providers.append(EchoProvider())
        elif name == "openai":
            providers.append(
                OpenAIProvider(
                    api_key=settings.openai_api_key or "",
                    base_url=settings.openai_base_url,
                    timeout_seconds=settings.openai_timeout_seconds,
                )
            )
        else:
            raise ConfigurationError(f"Unknown provider '{name}' in provider_chain.")

    return providers
