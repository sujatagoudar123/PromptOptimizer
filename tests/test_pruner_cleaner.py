"""Tests for the new token pruner and structural cleaner."""
from __future__ import annotations

import pytest

from llm_gateway.optimization.cleaner import StructuralCleaner
from llm_gateway.optimization.pruner import TokenPruner


class TestTokenPruner:
    def setup_method(self):
        self.p = TokenPruner()

    def test_removes_articles_and_stopwords(self):
        out, n = self.p.prune("Summarize the document and list the key points.")
        assert n >= 2  # "the" twice at least
        assert "the document" not in out.lower()

    def test_does_not_touch_constraints(self):
        text = "You must not skip steps. Also please do it quickly."
        out, _ = self.p.prune(text)
        # constraint sentence preserved verbatim
        assert "You must not skip steps." in out
        # non-constraint sentence loses filler
        assert "please" not in out.lower() or out.count("please") < text.count("please")

    def test_preserves_fenced_code(self):
        text = "Do this:\n```\nfor the x in the list: print(the x)\n```\nthanks"
        out, _ = self.p.prune(text)
        # The code block is untouched — "the x" stays
        assert "for the x in the list: print(the x)" in out

    def test_preserves_quoted_strings(self):
        text = 'Return a field named "the answer" with the value.'
        out, _ = self.p.prune(text)
        # quoted string untouched
        assert '"the answer"' in out

    def test_preserves_json(self):
        text = 'Return {"the_key": "the_value"} as the result.'
        out, _ = self.p.prune(text)
        assert '{"the_key": "the_value"}' in out

    def test_preserves_negators(self):
        text = "Do not mention the name."
        out, _ = self.p.prune(text)
        # constraint line — completely untouched
        assert out == text

    def test_idempotent(self):
        once, _ = self.p.prune("Summarize the document and list the key points.")
        twice, _ = self.p.prune(once)
        assert once == twice


class TestStructuralCleaner:
    def setup_method(self):
        self.c = StructuralCleaner()

    def test_minifies_fenced_json(self):
        text = (
            "Parse this:\n"
            "```json\n"
            "{\n"
            '  "name": "Alice",\n'
            '  "age":  30\n'
            "}\n"
            "```"
        )
        out, saved = self.c.clean(text)
        assert saved > 0
        assert '{"name":"Alice","age":30}' in out

    def test_collapses_blank_line_runs(self):
        text = "line 1\n\n\n\n\nline 2"
        out, _ = self.c.clean(text)
        assert "\n\n\n" not in out

    def test_strips_html_comments(self):
        text = "Hello <!-- this is a secret note --> world."
        out, _ = self.c.clean(text)
        assert "secret" not in out
        assert "Hello" in out and "world." in out

    def test_preserves_non_json_fences(self):
        text = "Code:\n```python\ndef f():\n    pass\n```"
        out, _ = self.c.clean(text)
        # Python fence left intact
        assert "def f():" in out
