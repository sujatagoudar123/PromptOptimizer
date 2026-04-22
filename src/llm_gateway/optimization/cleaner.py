"""Structural cleanup for RAG-style prompts containing pasted artifacts.

Real-world prompts often contain JSON blobs, HTML fragments, log lines,
and heavily indented text pasted verbatim. This cleaner reduces token
count safely by:

  * Minifying JSON (drop indentation + trailing whitespace) when it
    clearly parses as JSON.
  * Collapsing 3+ consecutive blank lines to 1 blank line.
  * Trimming trailing whitespace per line.
  * Removing HTML comments (`<!-- ... -->`).

It does NOT reorder, rewrite, or remove content — only removes whitespace
and formatting that carries no meaning.
"""
from __future__ import annotations

import json
import re


# Detect a fenced JSON block: ```json ... ``` or ```JSON ... ```
_FENCE_JSON_RE = re.compile(r"```(?:json|JSON)\s*\n(.*?)\n```", re.DOTALL)

# A standalone JSON object/array on its own line(s). Conservative: must
# start with { or [ at the start of a line, end on a matching close,
# and parse cleanly.
_JSON_OBJECT_RE = re.compile(r"(?m)^(\s*)(\{[\s\S]*?\}|\[[\s\S]*?\])\s*$")

# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# 3+ blank lines -> 1 blank line
_BLANK_RUN_RE = re.compile(r"\n\s*\n\s*\n+")


class StructuralCleaner:
    """Minify embedded JSON, drop HTML comments, collapse blank runs."""

    def clean(self, text: str) -> tuple[str, int]:
        """Return (cleaned_text, chars_saved)."""
        if not text:
            return text, 0

        original_len = len(text)

        # 1. Minify fenced JSON blocks
        def _minify_fenced(m: re.Match[str]) -> str:
            body = m.group(1)
            try:
                return "```json\n" + json.dumps(json.loads(body), separators=(",", ":")) + "\n```"
            except (ValueError, json.JSONDecodeError):
                return m.group(0)

        text = _FENCE_JSON_RE.sub(_minify_fenced, text)

        # 2. Minify standalone JSON blocks
        def _minify_standalone(m: re.Match[str]) -> str:
            indent, body = m.group(1), m.group(2)
            try:
                return indent + json.dumps(json.loads(body), separators=(",", ":"))
            except (ValueError, json.JSONDecodeError):
                return m.group(0)

        text = _JSON_OBJECT_RE.sub(_minify_standalone, text)

        # 3. Drop HTML comments
        text = _HTML_COMMENT_RE.sub("", text)

        # 4. Collapse runs of blank lines
        text = _BLANK_RUN_RE.sub("\n\n", text)

        # 5. Trim trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        saved = original_len - len(text)
        return text, max(0, saved)
