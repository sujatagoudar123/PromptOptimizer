"""Request governance: policy checks applied BEFORE optimization.

Responsibilities:
  * reject empty prompts
  * estimate token count
  * enforce max_prompt_tokens (reject or defer to optimizer to compress)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.exceptions import EmptyPromptError, OversizePromptError
from .tokens import TokenEstimator


@dataclass
class GovernanceResult:
    token_count: int
    warnings: list[str]


class GovernancePolicy:
    def __init__(
        self,
        estimator: TokenEstimator,
        *,
        max_prompt_tokens: int,
        reject_oversize: bool,
    ) -> None:
        self._estimator = estimator
        self._max_prompt_tokens = max_prompt_tokens
        self._reject_oversize = reject_oversize

    def evaluate(self, prompt: str) -> GovernanceResult:
        if prompt is None or not prompt.strip():
            raise EmptyPromptError("Prompt must be a non-empty string.")

        tokens = self._estimator.estimate(prompt)
        warnings: list[str] = []

        if tokens > self._max_prompt_tokens:
            if self._reject_oversize:
                raise OversizePromptError(
                    f"Prompt has {tokens} tokens, exceeds max {self._max_prompt_tokens}.",
                    detail="Set LLMGW_REJECT_OVERSIZE=false to enable auto-compression.",
                )
            warnings.append(
                f"Prompt is oversized ({tokens} > {self._max_prompt_tokens}); "
                "aggressive compression will be attempted."
            )

        return GovernanceResult(token_count=tokens, warnings=warnings)
