"""REST API routes.

Endpoints:
  POST /v1/complete   — main gateway endpoint
  GET  /health        — liveness/readiness
  GET  /metrics       — Prometheus exposition format
  GET  /stats         — JSON snapshot used by the dashboard
  GET  /              — dashboard HTML
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config.settings import Settings
from ..core.exceptions import GatewayError
from ..core.models import CompletionRequest, CompletionResponse, ErrorBody, HealthResponse
from ..core.orchestrator import Gateway
from ..observability.metrics import Metrics
from .dependencies import get_gateway, get_metrics, get_settings_dep
from .. import __version__


router = APIRouter()


# --- Recent-requests ring buffer (for dashboard /stats) -----------------

class RecentRequestsBuffer:
    """Small in-memory ring of the last N completions for the UI."""

    def __init__(self, size: int = 50) -> None:
        self._size = size
        self._buf: list[dict] = []

    def add(self, entry: dict) -> None:
        self._buf.append(entry)
        if len(self._buf) > self._size:
            self._buf = self._buf[-self._size :]

    def snapshot(self) -> list[dict]:
        return list(reversed(self._buf))


# --- Routes -------------------------------------------------------------


@router.post(
    "/v1/complete",
    response_model=CompletionResponse,
    responses={
        400: {"model": ErrorBody},
        413: {"model": ErrorBody},
        502: {"model": ErrorBody},
        503: {"model": ErrorBody},
        504: {"model": ErrorBody},
    },
)
async def complete(
    body: CompletionRequest,
    request: Request,
    gateway: Gateway = Depends(get_gateway),
) -> CompletionResponse:
    try:
        resp = await gateway.complete(body)
    except GatewayError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.code, "message": e.message, "detail": e.detail},
        )

    # Push summary into the dashboard buffer
    buf: RecentRequestsBuffer = request.app.state.recent_requests
    buf.add(
        {
            "request_id": resp.metadata.request_id,
            "original_prompt": resp.metadata.original_prompt[:200],
            "optimized_prompt": resp.metadata.optimized_prompt[:200],
            "tokens_saved": resp.metadata.tokens.tokens_saved,
            "original_tokens": resp.metadata.tokens.original_prompt_tokens,
            "optimized_tokens": resp.metadata.tokens.optimized_prompt_tokens,
            "latency_ms": resp.metadata.latency_ms,
            "cache_status": resp.metadata.cache_status.value,
            "routing_path": resp.metadata.routing_path.value,
            "provider": resp.metadata.provider,
            "model": resp.metadata.model,
            "techniques": resp.metadata.optimization.techniques,
            "warnings": resp.metadata.warnings,
            "prompt_quality_score": resp.metadata.prompt_quality.score,
            "coached": resp.metadata.prompt_quality.coached,
        }
    )
    return resp


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        providers=settings.provider_list,
        cache_backend=settings.cache_backend,
    )


@router.get("/metrics")
async def metrics(metrics_: Metrics = Depends(get_metrics)) -> Response:
    data = generate_latest(metrics_.registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@router.get("/stats")
async def stats(
    request: Request,
    gateway: Gateway = Depends(get_gateway),
    metrics_: Metrics = Depends(get_metrics),
) -> JSONResponse:
    """JSON snapshot consumed by the dashboard."""
    buf: RecentRequestsBuffer = request.app.state.recent_requests

    # Pull current counter values from the prometheus registry
    def _counter_value(counter, labels: dict | None = None) -> float:
        try:
            if labels:
                return counter.labels(**labels)._value.get()
            return counter._value.get()
        except Exception:
            return 0.0

    total_requests = _counter_value(metrics_.requests_total)
    exact_hits = _counter_value(metrics_.cache_hits_total, {"kind": "exact"})
    semantic_hits = _counter_value(metrics_.cache_hits_total, {"kind": "semantic"})
    misses = _counter_value(metrics_.cache_misses_total)
    tokens_saved = _counter_value(metrics_.tokens_saved_total)
    bypasses = _counter_value(metrics_.optimization_bypasses_total)

    total_cache_attempts = exact_hits + semantic_hits + misses
    exact_hit_rate = (exact_hits / total_cache_attempts) if total_cache_attempts else 0.0
    semantic_hit_rate = (semantic_hits / total_cache_attempts) if total_cache_attempts else 0.0
    miss_rate = (misses / total_cache_attempts) if total_cache_attempts else 0.0

    # Latency histogram summary
    latency_hist = metrics_.request_latency_seconds
    samples = list(latency_hist.collect())
    count = 0.0
    total = 0.0
    for metric in samples:
        for s in metric.samples:
            if s.name.endswith("_count"):
                count = s.value
            elif s.name.endswith("_sum"):
                total = s.value
    avg_latency_ms = (total / count * 1000.0) if count else 0.0

    return JSONResponse(
        {
            "totals": {
                "requests": int(total_requests),
                "tokens_saved": int(tokens_saved),
                "optimization_bypasses": int(bypasses),
                "cache_entries": await gateway._cache.size(),  # internal but fine here
            },
            "cache": {
                "exact_hits": int(exact_hits),
                "semantic_hits": int(semantic_hits),
                "misses": int(misses),
                "exact_hit_rate": round(exact_hit_rate, 4),
                "semantic_hit_rate": round(semantic_hit_rate, 4),
                "miss_rate": round(miss_rate, 4),
            },
            "latency": {
                "avg_ms": round(avg_latency_ms, 2),
            },
            "recent": buf.snapshot(),
        }
    )


# --- Dashboard ----------------------------------------------------------

UI_DIR = Path(__file__).resolve().parents[3] / "ui"


@router.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    index = UI_DIR / "index.html"
    return FileResponse(index)
