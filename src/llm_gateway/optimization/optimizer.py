"""Optimization pipeline.

Composes (in order):
    Cleaner       — minify embedded JSON, drop HTML comments, collapse blank runs
    Normalizer    — structural whitespace / Unicode cleanup
    Deduper       — remove repeated lines and sentences
    Compressor    — remove filler phrases (e.g. "could you please")
    TokenPruner   — drop low-information stopwords in safe contexts

then runs Guardrails to produce a confidence score and warnings. If
guardrails invalidate the result (confidence == 0), we fall back to the
normalized-but-not-compressed version to never regress on correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.models import OptimizationReport
from .cleaner import StructuralCleaner
from .compressor import Compressor
from .deduper import Deduper
from .guardrails import Guardrails
from .normalizer import Normalizer
from .pruner import TokenPruner


@dataclass
class OptimizationOutput:
    optimized_prompt: str
    report: OptimizationReport


class PromptOptimizer:
    def __init__(
        self,
        *,
        cleaner: StructuralCleaner | None = None,
        normalizer: Normalizer | None = None,
        deduper: Deduper | None = None,
        compressor: Compressor | None = None,
        pruner: TokenPruner | None = None,
        guardrails: Guardrails | None = None,
        enabled: bool = True,
        aggressive_pruning: bool = False,
    ) -> None:
        self._cleaner = cleaner or StructuralCleaner()
        self._normalizer = normalizer or Normalizer()
        self._deduper = deduper or Deduper()
        self._compressor = compressor or Compressor()
        self._pruner = pruner or TokenPruner()
        self._guardrails = guardrails or Guardrails()
        self._enabled = enabled
        self._aggressive_pruning = aggressive_pruning

    def optimize(self, prompt: str, *, bypass: bool = False) -> OptimizationOutput:
        if bypass or not self._enabled:
            return OptimizationOutput(
                optimized_prompt=prompt,
                report=OptimizationReport(
                    applied=False,
                    confidence=1.0,
                    techniques=[],
                    warnings=[],
                ),
            )

        techniques: list[str] = []

        # Step 1: structural cleanup (JSON minify, blank-line collapse)
        cleaned, chars_saved = self._cleaner.clean(prompt)
        if chars_saved > 0:
            techniques.append(f"clean({chars_saved}c)")

        # Step 2: whitespace / Unicode normalization
        normalized = self._normalizer.normalize(cleaned)
        if normalized != cleaned:
            techniques.append("normalize")

        # Step 3: dedupe repeated instructions
        deduped, dedupe_count = self._deduper.dedupe(normalized)
        if dedupe_count > 0:
            techniques.append(f"dedupe({dedupe_count})")

        # Step 4: safe compression (filler-phrase removal)
        compressed, compress_count = self._compressor.compress(deduped)
        if compress_count > 0:
            techniques.append(f"compress({compress_count})")

        # Step 5: entropy-based stopword pruning (aggressive, off by default
        # for per-request override, on by default at global level — this is
        # the "real" token savings lever)
        final_before_guardrails = compressed
        if self._aggressive_pruning:
            pruned, pruned_count = self._pruner.prune(compressed)
            if pruned_count > 0:
                techniques.append(f"prune({pruned_count})")
            final_before_guardrails = pruned

        # Guardrails: validate confidence against the *original*
        confidence, warnings = self._guardrails.evaluate(
            prompt, final_before_guardrails
        )

        # If guardrails say the optimized prompt is unusable, fall back.
        final = final_before_guardrails
        if confidence == 0.0:
            final = normalized if normalized.strip() else prompt
            techniques = [t for t in techniques if t == "normalize"]

        return OptimizationOutput(
            optimized_prompt=final,
            report=OptimizationReport(
                applied=bool(techniques),
                confidence=confidence,
                techniques=techniques,
                warnings=warnings,
            ),
        )
