"""Application entrypoint and dependency wiring.

Usage:
    python -m llm_gateway.main       # run uvicorn with defaults
    uvicorn llm_gateway.main:app     # run under an external ASGI server
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .api.routes import RecentRequestsBuffer, router
from .caching.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .caching.manager import CacheManager
from .caching.memory import MemoryCache
from .caching.base import CacheBackend
from .coaching.coach import PromptCoach
from .coaching.scorer import PromptQualityScorer
from .config.settings import Settings, get_settings
from .core.orchestrator import Gateway
from .governance.policy import GovernancePolicy
from .governance.tokens import build_estimator
from .observability.logging import configure_logging, get_logger
from .observability.metrics import Metrics
from .optimization.optimizer import PromptOptimizer
from .providers.registry import build_providers
from .routing.router import Router


log = get_logger(__name__)


def _build_cache_backend(settings: Settings) -> CacheBackend:
    if settings.cache_backend == "redis":
        # Imported lazily so the package isn't required for memory mode.
        from .caching.redis_backend import RedisCache

        return RedisCache(
            settings.redis_url,
            ttl_seconds=settings.cache_ttl_seconds,
        )
    return MemoryCache(
        max_entries=settings.cache_max_entries,
        ttl_seconds=settings.cache_ttl_seconds,
    )


def _build_embedder(settings: Settings) -> Embedder | None:
    if not settings.semantic_cache_enabled:
        return None
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            log.warning(
                "embedder.openai_missing_key",
                detail="Falling back to HashingEmbedder.",
            )
            return HashingEmbedder(dim=settings.embedding_dim)
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    return HashingEmbedder(dim=settings.embedding_dim)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    log.info(
        "gateway.starting",
        version=__version__,
        environment=settings.environment,
        providers=settings.provider_list,
        cache=settings.cache_backend,
    )

    metrics = Metrics()
    token_estimator = build_estimator()
    governance = GovernancePolicy(
        token_estimator,
        max_prompt_tokens=settings.max_prompt_tokens,
        reject_oversize=settings.reject_oversize,
    )
    optimizer = PromptOptimizer(
        enabled=settings.optimization_enabled,
        aggressive_pruning=settings.aggressive_pruning,
    )

    cache_backend = _build_cache_backend(settings)
    embedder = _build_embedder(settings)
    cache = CacheManager(
        backend=cache_backend,
        embedder=embedder,
        semantic_enabled=settings.semantic_cache_enabled,
        similarity_threshold=settings.semantic_similarity_threshold,
        metrics=metrics,
    )

    providers = build_providers(settings)
    router_ = Router(providers, metrics=metrics)

    gateway = Gateway(
        scorer=PromptQualityScorer(),
        coach=PromptCoach(),
        optimizer=optimizer,
        cache=cache,
        router=router_,
        governance=governance,
        token_estimator=token_estimator,
        metrics=metrics,
        default_model=settings.default_model,
        coaching_enabled=settings.coaching_enabled,
        prompt_quality_threshold=settings.prompt_quality_threshold,
    )

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        yield
        await gateway.aclose()
        log.info("gateway.stopped")

    app = FastAPI(
        title="LLM Optimization Gateway",
        version=__version__,
        description="Middleware gateway for LLM providers: optimization, "
        "caching, routing, governance, observability.",
        lifespan=lifespan,
    )

    # Dependency-accessible singletons
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.gateway = gateway
    app.state.recent_requests = RecentRequestsBuffer(size=50)

    app.include_router(router)

    return app


# Module-level app for `uvicorn llm_gateway.main:app`
app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "llm_gateway.main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    sys.exit(main())
