"""Token estimation.

Uses tiktoken when available for accurate counts with the cl100k_base
encoding (GPT-3.5/4 family). Falls back to a character-heuristic so
the gateway still functions in locked-down environments.
"""
from __future__ import annotations

from typing import Protocol


class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...


class TiktokenEstimator:
    """Backed by tiktoken's cl100k_base. Accurate for modern OpenAI models."""

    def __init__(self) -> None:
        import tiktoken  # lazy import

        self._enc = tiktoken.get_encoding("cl100k_base")

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


class HeuristicEstimator:
    """Rough char-based fallback: ~4 chars per token (English avg).

    Overestimates slightly on whitespace-heavy prompts, which is the
    safer side for governance decisions.
    """

    AVG_CHARS_PER_TOKEN = 4

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        # Use max(1, ...) so any non-empty string counts as at least 1 token.
        return max(1, (len(text) + self.AVG_CHARS_PER_TOKEN - 1) // self.AVG_CHARS_PER_TOKEN)


def build_estimator() -> TokenEstimator:
    """Return the best available estimator on this system."""
    try:
        return TiktokenEstimator()
    except Exception:  # pragma: no cover - environment dependent
        return HeuristicEstimator()
