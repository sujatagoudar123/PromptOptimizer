"""Governance layer tests."""
from __future__ import annotations

import pytest

from llm_gateway.core.exceptions import EmptyPromptError, OversizePromptError
from llm_gateway.governance.policy import GovernancePolicy
from llm_gateway.governance.tokens import HeuristicEstimator, build_estimator


def test_heuristic_estimator_nonempty():
    e = HeuristicEstimator()
    assert e.estimate("") == 0
    assert e.estimate("a") == 1
    assert e.estimate("a" * 20) == 5


def test_build_estimator_returns_something():
    e = build_estimator()
    assert e.estimate("hello world") > 0


def test_empty_prompt_raises():
    p = GovernancePolicy(
        build_estimator(), max_prompt_tokens=1000, reject_oversize=False
    )
    with pytest.raises(EmptyPromptError):
        p.evaluate("")
    with pytest.raises(EmptyPromptError):
        p.evaluate("   \n\t  ")


def test_oversize_rejected_when_configured():
    p = GovernancePolicy(
        build_estimator(), max_prompt_tokens=5, reject_oversize=True
    )
    with pytest.raises(OversizePromptError):
        p.evaluate("this prompt is definitely way longer than five tokens should ever be")


def test_oversize_warned_when_not_rejected():
    p = GovernancePolicy(
        build_estimator(), max_prompt_tokens=5, reject_oversize=False
    )
    result = p.evaluate("this prompt is way longer than five tokens")
    assert result.token_count > 5
    assert any("oversized" in w.lower() for w in result.warnings)


def test_within_limits_no_warnings():
    p = GovernancePolicy(
        build_estimator(), max_prompt_tokens=1000, reject_oversize=False
    )
    result = p.evaluate("short prompt")
    assert result.warnings == []
    assert result.token_count > 0
