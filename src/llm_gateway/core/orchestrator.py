"""Gateway orchestrator.

Implements the full request pipeline:

    governance -> optimize -> cache lookup -> route -> cache store -> metrics

This is the single entry point for all completion requests. The API layer
is a thin wrapper around ``Gateway.complete``.
"""
from __future__ import annotations

import time
import uuid

from ..caching.manager import CacheManager
from ..coaching.coach import PromptCoach
from ..coaching.scorer import PromptQualityScorer
from ..core.exceptions import GatewayError
from ..core.models import (
    CacheStatus,
    CompletionMetadata,
    CompletionRequest,
    CompletionResponse,
    DimensionScoreModel,
    PromptQualityReport,
    ProviderRequest,
    RoutingPath,
    TokenStats,
)
from ..governance.policy import GovernancePolicy
from ..governance.tokens import TokenEstimator
from ..observability.logging import bind_request_context, clear_request_context, get_logger
from ..observability.metrics import Metrics
from ..optimization.optimizer import PromptOptimizer
from ..routing.router import Router


log = get_logger(__name__)


class Gateway:
    def __init__(
        self,
        *,
        scorer: PromptQualityScorer,
        coach: PromptCoach,
        optimizer: PromptOptimizer,
        cache: CacheManager,
        router: Router,
        governance: GovernancePolicy,
        token_estimator: TokenEstimator,
        metrics: Metrics,
        default_model: str,
        coaching_enabled: bool,
        prompt_quality_threshold: float,
    ) -> None:
        self._scorer = scorer
        self._coach = coach
        self._optimizer = optimizer
        self._cache = cache
        self._router = router
        self._governance = governance
        self._tokens = token_estimator
        self._metrics = metrics
        self._default_model = default_model
        self._coaching_enabled = coaching_enabled
        self._quality_threshold = prompt_quality_threshold

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        request_id = str(uuid.uuid4())
        bind_request_context(request_id=request_id)
        self._metrics.requests_total.inc()
        started = time.perf_counter()

        try:
            return await self._complete_inner(req, request_id, started)
        except GatewayError as e:
            self._metrics.record_error(e.code)
            log.warning("gateway.error", code=e.code, message=e.message)
            raise
        except Exception as e:
            self._metrics.record_error("unhandled")
            log.error("gateway.unhandled_error", error=str(e))
            raise
        finally:
            clear_request_context()

    async def _complete_inner(
        self, req: CompletionRequest, request_id: str, started: float
    ) -> CompletionResponse:
        model = req.model or self._default_model
        warnings: list[str] = []

        # --- 1. Governance ---
        gov = self._governance.evaluate(req.prompt)
        warnings.extend(gov.warnings)
        original_tokens = gov.token_count

        # --- 2. Prompt Coaching (quality scoring + conditional rewrite) ---
        # Always score. Only rewrite if below threshold and not bypassed.
        quality = self._scorer.score(req.prompt)
        coach_apply = self._coaching_enabled and not req.bypass_coaching
        if coach_apply:
            coaching = self._coach.coach(
                req.prompt,
                quality,
                score_threshold=self._quality_threshold,
            )
        else:
            # Build a no-op coaching result so downstream code is uniform.
            from ..coaching.coach import CoachingResult
            coaching = CoachingResult(
                rewritten_prompt=req.prompt,
                applied=False,
                techniques_applied=[],
                reasoning=[],
                suggestions=[],
            )

        coached_prompt = coaching.rewritten_prompt
        if coaching.applied:
            self._metrics.coaching_applied_total.inc()
            log.info(
                "gateway.coached",
                score=quality.score,
                techniques=coaching.techniques_applied,
            )

        prompt_quality_report = PromptQualityReport(
            score=quality.score,
            threshold=self._quality_threshold,
            dimensions=[
                DimensionScoreModel(
                    name=d.name, score=round(d.score, 4),
                    weight=d.weight, reason=d.reason,
                )
                for d in quality.dimensions
            ],
            coached=coaching.applied,
            techniques_applied=coaching.techniques_applied,
            reasoning=coaching.reasoning,
            suggestions=coaching.suggestions,
            strengths=quality.strengths,
            weaknesses=quality.weaknesses,
        )

        # --- 3. Optimization (on the coached prompt) ---
        if req.bypass_optimization:
            self._metrics.optimization_bypasses_total.inc()
        opt = self._optimizer.optimize(coached_prompt, bypass=req.bypass_optimization)
        optimized_prompt = opt.optimized_prompt
        optimized_tokens = self._tokens.estimate(optimized_prompt)
        tokens_saved = max(0, original_tokens - optimized_tokens)
        self._metrics.tokens_saved_total.inc(tokens_saved)

        warnings.extend(opt.report.warnings)

        savings_ratio = (
            tokens_saved / original_tokens if original_tokens > 0 else 0.0
        )
        log.info(
            "gateway.optimized",
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            saved=tokens_saved,
            confidence=opt.report.confidence,
            techniques=opt.report.techniques,
        )

        # --- 4. Cache lookup ---
        cache_status = CacheStatus.BYPASS if req.bypass_cache else CacheStatus.MISS
        routing_path = RoutingPath.UPSTREAM
        completion_text: str | None = None
        provider_name = "cache"
        completion_tokens: int | None = None

        if not req.bypass_cache:
            lookup = await self._cache.lookup(optimized_prompt, model)
            cache_status = lookup.status
            if lookup.value is not None:
                completion_text = lookup.value
                routing_path = RoutingPath.CACHE
                provider_name = "cache"

        # --- 5. Upstream routing (on cache miss or bypass) ---
        if completion_text is None:
            route_result = await self._router.complete(
                ProviderRequest(
                    prompt=optimized_prompt,
                    model=model,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
            )
            completion_text = route_result.response.completion
            completion_tokens = route_result.response.completion_tokens
            provider_name = route_result.response.provider
            model = route_result.response.model
            routing_path = route_result.path

            # --- 6. Store in cache ---
            if not req.bypass_cache:
                await self._cache.store(
                    optimized_prompt,
                    model,
                    completion_text,
                    unsafe_to_cache=req.unsafe_to_cache,
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._metrics.request_latency_seconds.observe(elapsed_ms / 1000.0)

        metadata = CompletionMetadata(
            original_prompt=req.prompt,
            coached_prompt=coached_prompt,
            optimized_prompt=optimized_prompt,
            tokens=TokenStats(
                original_prompt_tokens=original_tokens,
                optimized_prompt_tokens=optimized_tokens,
                completion_tokens=completion_tokens,
                tokens_saved=tokens_saved,
                savings_ratio=round(savings_ratio, 4),
            ),
            latency_ms=round(elapsed_ms, 2),
            cache_status=cache_status,
            routing_path=routing_path,
            provider=provider_name,
            model=model,
            prompt_quality=prompt_quality_report,
            optimization=opt.report,
            warnings=warnings,
            request_id=request_id,
        )

        return CompletionResponse(completion=completion_text, metadata=metadata)

    async def aclose(self) -> None:
        await self._router.aclose()
