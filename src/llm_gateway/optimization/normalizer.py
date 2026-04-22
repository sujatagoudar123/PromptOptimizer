"""Prompt normalization.

Structural cleanup that is *always* safe:
  * collapse repeated whitespace
  * strip per-line leading/trailing whitespace
  * collapse 3+ consecutive newlines to 2 (preserve paragraphs)
  * strip zero-width and non-printing chars
  * normalize smart quotes/dashes to ASCII for stable hashing
"""
from __future__ import annotations

import re
import unicodedata


# Matches 3+ consecutive newlines (with optional whitespace between) -> "\n\n"
_MULTI_NEWLINE_RE = re.compile(r"(?:[ \t]*\n[ \t]*){3,}")
# Matches runs of spaces/tabs (but NOT newlines)
_INLINE_WS_RE = re.compile(r"[ \t]{2,}")
# Zero-width and control characters (keep \n and \t)
_CTRL_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF\x00-\x08\x0B\x0C\x0E-\x1F]")

_SMART_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
    "\u00A0": " ",  # non-breaking space
}


class Normalizer:
    """Idempotent, lossless-of-intent structural cleanup."""

    def normalize(self, text: str) -> str:
        if not text:
            return text

        # Unicode NFC so "é" vs "e + combining acute" hash identically.
        text = unicodedata.normalize("NFC", text)

        # Replace smart punctuation before anything else
        for src, dst in _SMART_QUOTES.items():
            if src in text:
                text = text.replace(src, dst)

        # Strip invisible/control chars
        text = _CTRL_RE.sub("", text)

        # Normalize Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse inline whitespace runs (but preserve single spaces and newlines)
        text = _INLINE_WS_RE.sub(" ", text)

        # Collapse excessive blank lines
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

        # Per-line trim
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text.strip()
