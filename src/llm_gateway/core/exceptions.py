"""Gateway exception hierarchy.

All recoverable errors in the request pipeline inherit from GatewayError
so the API layer can map them to appropriate HTTP status codes.
"""
from __future__ import annotations


class GatewayError(Exception):
    """Base class for all gateway-originated errors."""

    status_code: int = 500
    code: str = "gateway_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class EmptyPromptError(GatewayError):
    status_code = 400
    code = "empty_prompt"


class OversizePromptError(GatewayError):
    status_code = 413
    code = "oversize_prompt"


class ProviderError(GatewayError):
    status_code = 502
    code = "provider_error"


class ProviderTimeoutError(ProviderError):
    status_code = 504
    code = "provider_timeout"


class AllProvidersFailedError(GatewayError):
    status_code = 503
    code = "all_providers_failed"


class CacheBackendError(GatewayError):
    """Non-fatal — the pipeline should continue on cache miss."""

    status_code = 500
    code = "cache_backend_error"


class ConfigurationError(GatewayError):
    status_code = 500
    code = "configuration_error"
