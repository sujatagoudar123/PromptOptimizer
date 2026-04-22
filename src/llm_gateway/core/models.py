"""Request and response data models used across the gateway."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CacheStatus(str, Enum):
    EXACT_HIT = "exact_hit"
    SEMANTIC_HIT = "semantic_hit"
    MISS = "miss"
    BYPASS = "bypass"


class RoutingPath(str, Enum):
    CACHE = "cache"
    UPSTREAM = "upstream"
    FAILOVER = "failover"


class CompletionRequest(BaseModel):
    """Input to the gateway."""

    prompt: str = Field(..., description="Raw user prompt.")
    model: str | None = Field(None, description="Override the default model.")
    max_tokens: int | None = Field(None, ge=1, le=32000)
    temperature: float | None = Field(None, ge=0.0, le=2.0)

    # Per-request overrides
    bypass_coaching: bool = False
    bypass_optimization: bool = False
    bypass_cache: bool = False
    unsafe_to_cache: bool = Field(
        False,
        description="Set true for prompts with volatile context (timestamps, "
        "user-specific secrets) that should never be semantically cached.",
    )

    # Freeform tags forwarded to structured logs / metrics labels.
    tags: dict[str, str] = Field(default_factory=dict)


class TokenStats(BaseModel):
    original_prompt_tokens: int
    optimized_prompt_tokens: int
    completion_tokens: int | None = None
    tokens_saved: int  # original - optimized (>= 0)
    savings_ratio: float  # 0.0-1.0


class OptimizationReport(BaseModel):
    applied: bool
    confidence: float  # 0.0-1.0
    techniques: list[str]  # e.g. ["normalize", "dedupe", "compress"]
    warnings: list[str]


class DimensionScoreModel(BaseModel):
    name: str
    score: float
    weight: float
    reason: str


class PromptQualityReport(BaseModel):
    """Prompt-engineering quality scoring and coaching output."""

    score: float  # composite 0.0 - 1.0
    threshold: float  # the threshold used to decide whether to rewrite
    dimensions: list[DimensionScoreModel]
    coached: bool  # true if the prompt was rewritten by the Coach
    techniques_applied: list[str]  # structural rewrites applied
    reasoning: list[str]  # human-readable explanation of each rewrite
    suggestions: list[str]  # hints the user should apply manually
    strengths: list[str]
    weaknesses: list[str]


class CompletionMetadata(BaseModel):
    original_prompt: str
    coached_prompt: str  # after the Coach (may equal original if above threshold)
    optimized_prompt: str  # after the Optimizer
    tokens: TokenStats
    latency_ms: float
    cache_status: CacheStatus
    routing_path: RoutingPath
    provider: str
    model: str
    prompt_quality: PromptQualityReport
    optimization: OptimizationReport
    warnings: list[str] = Field(default_factory=list)
    request_id: str


class CompletionResponse(BaseModel):
    completion: str
    metadata: CompletionMetadata


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    providers: list[str]
    cache_backend: str


class ErrorBody(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


# Internal types (not serialized over the wire but shared between layers)

class ProviderRequest(BaseModel):
    prompt: str
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    completion: str
    provider: str
    model: str
    completion_tokens: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
