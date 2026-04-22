"""Structured JSON logging using structlog.

All gateway components should obtain a logger via ``get_logger(__name__)``.
Log lines include: timestamp, level, logger name, event, and any
keyword context the caller binds (request_id, provider, etc).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotent logging configuration. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


def bind_request_context(**kwargs: Any) -> None:
    """Bind key/value pairs to all log lines for the current request."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
