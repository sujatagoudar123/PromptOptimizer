"""Optimization guardrails.

After each optimization pass we score how confident we are that meaning
is preserved and emit warnings for anything suspicious.
"""
from __future__ import annotations

import re


# Intent-bearing signals whose presence/count must not change after optimization.
_INTENT_SIGNALS = [
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b|\bdon'?t\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\brequired\b", re.IGNORECASE),
    re.compile(r"\bformat:\b", re.IGNORECASE),
    re.compile(r"```"),  # fenced code
    re.compile(r"\{[^{}]*\}"),  # JSON-ish braces (count-stable is a sanity check)
]


class Guardrails:
    """Compute a confidence score and warnings for an optimization."""

    def __init__(self, *, confidence_warn_threshold: float = 0.6) -> None:
        self._warn_threshold = confidence_warn_threshold

    def evaluate(
        self, original: str, optimized: str
    ) -> tuple[float, list[str]]:
        """Return (confidence in [0,1], warnings)."""
        warnings: list[str] = []

        if not original:
            return 1.0, warnings

        # 1. Intent-signal counts must match exactly.
        mismatches: list[str] = []
        for pat in _INTENT_SIGNALS:
            a, b = len(pat.findall(original)), len(pat.findall(optimized))
            if a != b:
                mismatches.append(f"{pat.pattern}: {a}->{b}")

        intent_penalty = min(0.5, 0.1 * len(mismatches))
        if mismatches:
            warnings.append(
                "Intent signal count changed after optimization: "
                + ", ".join(mismatches)
            )

        # 2. Over-compression penalty: if we cut more than 70% of the text,
        # that's suspiciously aggressive and might have removed meaning.
        orig_len = len(original.strip())
        opt_len = len(optimized.strip())
        if orig_len > 0:
            reduction = 1.0 - (opt_len / orig_len)
        else:
            reduction = 0.0

        over_compression_penalty = 0.0
        if reduction > 0.7:
            over_compression_penalty = min(0.3, (reduction - 0.7) * 2.0)
            warnings.append(
                f"Optimization removed {reduction:.0%} of text; verify output quality."
            )

        # 3. If the optimized prompt ended up empty, that's a fatal guardrail trip.
        if not optimized.strip() and original.strip():
            warnings.append("Optimization produced empty output; falling back to original.")
            return 0.0, warnings

        confidence = max(0.0, 1.0 - intent_penalty - over_compression_penalty)

        if confidence < self._warn_threshold:
            warnings.append(
                f"Optimization confidence {confidence:.2f} is below threshold "
                f"{self._warn_threshold:.2f}."
            )

        return confidence, warnings
