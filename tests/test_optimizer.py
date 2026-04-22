"""Unit tests for the optimization layer."""
from __future__ import annotations

import pytest

from llm_gateway.optimization.compressor import Compressor
from llm_gateway.optimization.deduper import Deduper
from llm_gateway.optimization.guardrails import Guardrails
from llm_gateway.optimization.normalizer import Normalizer
from llm_gateway.optimization.optimizer import PromptOptimizer


class TestNormalizer:
    def test_collapses_whitespace(self):
        n = Normalizer()
        out = n.normalize("hello    world\t\t!")
        assert out == "hello world !"

    def test_collapses_multiple_newlines(self):
        n = Normalizer()
        out = n.normalize("a\n\n\n\n\nb")
        assert out == "a\n\nb"

    def test_strips_zero_width(self):
        n = Normalizer()
        out = n.normalize("hel\u200blo")
        assert out == "hello"

    def test_smart_quotes_normalized(self):
        n = Normalizer()
        out = n.normalize("\u201csmart\u201d and \u2018quotes\u2019")
        assert out == '"smart" and \'quotes\''

    def test_empty(self):
        n = Normalizer()
        assert n.normalize("") == ""


class TestDeduper:
    def test_removes_duplicate_lines(self):
        d = Deduper()
        out, removed = d.dedupe("Be concise.\nDo the work.\nBe concise.")
        assert removed == 1
        assert out.count("Be concise.") == 1

    def test_preserves_order(self):
        d = Deduper()
        out, _ = d.dedupe("First line here.\nSecond line here.\nFirst line here.")
        lines = out.split("\n")
        assert lines[0] == "First line here."
        assert lines[1] == "Second line here."

    def test_short_lines_not_deduped(self):
        d = Deduper(min_len=8)
        out, removed = d.dedupe("- a\n- b\n- a")
        assert removed == 0


class TestCompressor:
    def test_removes_please_kindly(self):
        c = Compressor()
        out, n = c.compress("Could you please summarize this document.")
        assert n >= 1
        assert "could you please" not in out.lower()

    def test_replaces_in_order_to(self):
        c = Compressor()
        out, _ = c.compress("I use this in order to save time.")
        assert "in order to" not in out
        assert " to " in out

    def test_preserves_code_blocks(self):
        c = Compressor()
        text = "please kindly fix:\n```\ndef f(): please kindly return 1\n```"
        out, _ = c.compress(text)
        # code block content stays verbatim
        assert "please kindly return 1" in out
        # outside-code filler is removed
        assert "please kindly fix" not in out.lower()

    def test_preserves_constraint_sentences(self):
        c = Compressor()
        # Two sentences: one constraint, one filler. Only the filler sentence
        # should be compressed; the constraint sentence is preserved verbatim.
        text = "You must not skip steps. Could you please summarize this."
        out, _ = c.compress(text)
        assert "You must not skip steps." in out
        assert "could you please" not in out.lower()

    def test_single_constraint_word_does_not_block_full_line(self):
        """Regression: 'exactly' in one sentence must not exempt adjacent
        sentences from compression."""
        c = Compressor()
        text = (
            "Summarize the report in exactly 3 bullets. "
            "Basically, I would like you to be concise."
        )
        out, n = c.compress(text)
        # The constraint sentence is preserved
        assert "exactly 3 bullets" in out
        # The other sentence is compressed
        assert "basically" not in out.lower()
        assert "i would like you to" not in out.lower()
        assert n >= 2


class TestGuardrails:
    def test_confidence_high_for_safe_compression(self):
        g = Guardrails()
        conf, warnings = g.evaluate(
            "Could you please summarize this.",
            "summarize this.",
        )
        assert conf > 0.8
        assert warnings == [] or all("confidence" not in w for w in warnings)

    def test_warns_on_intent_loss(self):
        g = Guardrails()
        # Original has "must not"; compressed drops it.
        conf, warnings = g.evaluate(
            "You must not mention price.",
            "mention price.",
        )
        assert any("Intent signal" in w for w in warnings)
        assert conf < 1.0

    def test_warns_on_over_compression(self):
        g = Guardrails()
        original = "a" * 100
        optimized = "a" * 10
        conf, warnings = g.evaluate(original, optimized)
        assert any("removed" in w.lower() for w in warnings)

    def test_empty_result_yields_zero_confidence(self):
        g = Guardrails()
        conf, warnings = g.evaluate("say hello", "")
        assert conf == 0.0


class TestPromptOptimizer:
    def test_bypass_returns_original(self):
        opt = PromptOptimizer()
        out = opt.optimize("Please kindly do X.", bypass=True)
        assert out.optimized_prompt == "Please kindly do X."
        assert out.report.applied is False

    def test_optimization_reduces_length(self):
        opt = PromptOptimizer()
        out = opt.optimize(
            "Could you please kindly summarize this document for me. "
            "I would like you to do that as soon as possible."
        )
        assert out.optimized_prompt != "Could you please kindly summarize this document for me. I would like you to do that as soon as possible."
        assert len(out.optimized_prompt) < len("Could you please kindly summarize this document for me. I would like you to do that as soon as possible.")
        assert out.report.applied is True
        assert "compress" in "".join(out.report.techniques)

    def test_falls_back_when_confidence_zero(self):
        """If compression would produce empty output, we fall back to normalized."""
        opt = PromptOptimizer()
        out = opt.optimize("please kindly")  # nothing left after compression
        # Must not be empty — guardrail should force a fallback
        assert out.optimized_prompt != ""
