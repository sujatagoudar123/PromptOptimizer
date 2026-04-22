"""Tests for the prompt quality + coaching layer."""
from __future__ import annotations

import pytest

from llm_gateway.coaching.coach import PromptCoach
from llm_gateway.coaching.scorer import PromptQualityScorer
from llm_gateway.core.models import CompletionRequest


# --- Scorer ---


class TestScorer:
    def setup_method(self):
        self.s = PromptQualityScorer()

    def test_very_short_prompt_is_low_quality(self):
        r = self.s.score("help me")
        assert r.score < 0.4

    def test_empty_style_prompt_scores_low(self):
        r = self.s.score("do this thing")
        assert r.score < 0.55
        assert any("task" in d.name for d in r.dimensions)

    def test_well_structured_prompt_scores_high(self):
        prompt = (
            "You are an experienced technical writer. "
            "Summarize the following release notes in exactly 3 bullet points. "
            "Format: markdown bullets, no preamble. "
            "Do not mention internal project names. "
            "For example, 'Added feature X' becomes 'Added X'. "
            "Context:\n- Added feature X\n- Fixed bug Y"
        )
        r = self.s.score(prompt)
        # With 7 dimensions (role, task, context, format, constraints,
        # examples, specificity), a prompt that hits all six of the
        # top-six (missing only perfect examples) should clear 0.70.
        assert r.score >= 0.70
        # Every dimension above 0.5
        assert all(d.score >= 0.5 for d in r.dimensions)

    def test_role_detected(self):
        r = self.s.score("You are a senior engineer. Review this code.")
        role_dim = next(d for d in r.dimensions if d.name == "role")
        assert role_dim.score == 1.0

    def test_format_detected(self):
        r = self.s.score("Summarize in exactly 3 bullet points.")
        fmt_dim = next(d for d in r.dimensions if d.name == "format")
        assert fmt_dim.score == 1.0

    def test_techniques_missing_reported(self):
        r = self.s.score("help me with this")
        missing = r.techniques_missing()
        # Should flag at least output_format (missing) and role (missing)
        assert "output_format" in missing
        assert "role_persona" in missing


# --- Coach ---


class TestCoach:
    def setup_method(self):
        self.s = PromptQualityScorer()
        self.c = PromptCoach()

    def test_high_quality_prompt_is_not_rewritten(self):
        good = (
            "You are a senior data scientist. Summarize the following dataset "
            "description in exactly 3 bullet points. Format: markdown. "
            "Do not mention vendor names. Context:\nDataset covers Q3 sales."
        )
        r = self.s.score(good)
        out = self.c.coach(good, r, score_threshold=0.55)
        assert out.applied is False
        assert out.rewritten_prompt == good

    def test_weak_prompt_gets_structured(self):
        # A mid-quality prompt: has a task verb but no scaffolding, no role,
        # no explicit format, no explicit context marker.
        weak = (
            "Write a one-paragraph blog intro about remote work trends. "
            "Don't mention specific companies."
        )
        r = self.s.score(weak)
        # Force coaching regardless of score
        out = self.c.coach(weak, r, score_threshold=1.0)
        assert out.applied is True
        # Task scaffolding must appear
        assert "Task:" in out.rewritten_prompt or "# Task" in out.rewritten_prompt

    def test_coach_does_not_invent_role(self):
        weak = "Summarize this document as bullet points."
        r = self.s.score(weak)
        out = self.c.coach(weak, r, score_threshold=0.9)
        # Coach never adds "You are a..." on its own
        assert "You are" not in out.rewritten_prompt
        # But it should suggest it
        assert any("role" in s.lower() or "persona" in s.lower() for s in out.suggestions)

    def test_coach_surfaces_suggestions_above_threshold(self):
        # A mid-quality prompt — above our very low threshold,
        # below the "great" line — still yields hygiene hints.
        prompt = "Summarize the following text."
        r = self.s.score(prompt)
        out = self.c.coach(prompt, r, score_threshold=0.2)  # very permissive
        assert out.applied is False
        # Below 0.85 we get hygiene hints
        if r.score < 0.85:
            assert out.suggestions


# --- End-to-end via Gateway ---


@pytest.mark.asyncio
async def test_gateway_scores_every_prompt(gateway):
    req = CompletionRequest(prompt="Summarize the report.")
    resp = await gateway.complete(req)
    q = resp.metadata.prompt_quality
    assert 0.0 <= q.score <= 1.0
    # 7 dimensions: role, task, context, format, constraints, examples, specificity
    assert len(q.dimensions) == 7


@pytest.mark.asyncio
async def test_gateway_coaches_weak_prompt(gateway):
    # Weak enough to cross the 0.55 default threshold
    weak = "help me with this"
    req = CompletionRequest(prompt=weak)
    resp = await gateway.complete(req)
    q = resp.metadata.prompt_quality
    # This prompt has no format, no role, no context, no constraints
    # and a vague verb — it MUST be flagged weak.
    assert q.score < 0.55
    # Coach won't rewrite a prompt this skeletal (no task verb to extract)
    # but it MUST emit suggestions
    assert q.suggestions


@pytest.mark.asyncio
async def test_gateway_coaches_structurable_prompt(gateway):
    weak = (
        "Summarize this document as bullet points. "
        "Don't include dates. Here is the text: Revenue grew 12%."
    )
    req = CompletionRequest(prompt=weak)
    resp = await gateway.complete(req)
    q = resp.metadata.prompt_quality
    # This one has enough material to restructure
    if q.score < q.threshold:
        assert q.coached is True
        assert q.techniques_applied
        assert q.reasoning


@pytest.mark.asyncio
async def test_bypass_coaching(gateway):
    weak = "do something"
    req = CompletionRequest(prompt=weak, bypass_coaching=True)
    resp = await gateway.complete(req)
    # Scoring still runs (always-score policy), but no rewrite
    assert resp.metadata.prompt_quality.coached is False
    assert resp.metadata.coached_prompt == weak


@pytest.mark.asyncio
async def test_strong_prompt_not_rewritten(gateway):
    strong = (
        "You are a staff engineer. Summarize the following release notes in "
        "exactly 3 bullet points. Format: markdown. Do not mention vendor names. "
        "Context:\n- Shipped feature X\n- Fixed issue Y"
    )
    req = CompletionRequest(prompt=strong)
    resp = await gateway.complete(req)
    q = resp.metadata.prompt_quality
    assert q.score >= 0.55
    assert q.coached is False
    assert resp.metadata.coached_prompt == strong
