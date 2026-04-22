"""FastAPI dependency providers.

These expose the singletons built during app startup to the route handlers.
They are set by ``main.create_app`` via ``app.state``.
"""
from __future__ import annotations

from fastapi import Depends, Request

from ..core.orchestrator import Gateway
from ..observability.metrics import Metrics
from ..config.settings import Settings


def get_gateway(request: Request) -> Gateway:
    return request.app.state.gateway  # type: ignore[no-any-return]


def get_metrics(request: Request) -> Metrics:
    return request.app.state.metrics  # type: ignore[no-any-return]


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


__all__ = ["get_gateway", "get_metrics", "get_settings_dep", "Depends"]
