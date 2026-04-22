"""Instruction-level deduplication.

Removes repeated boilerplate instructions while preserving the original
order of first occurrence. Operates line-by-line and sentence-by-sentence
within a line.

Critically it is *case and punctuation insensitive* for comparison but
preserves the original casing/punctuation of the first occurrence.
"""
from __future__ import annotations

import re


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _canonical(s: str) -> str:
    """Canonical key for dedupe comparison: lowercase, alnum+space only."""
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


class Deduper:
    """Dedup repeated lines and sentences, preserving order."""

    def __init__(self, *, min_len: int = 8) -> None:
        # Don't dedupe extremely short lines — they are more likely
        # to be meaningful separators or list items.
        self._min_len = min_len

    def dedupe(self, text: str) -> tuple[str, int]:
        """Return (deduped_text, number_of_segments_removed)."""
        if not text:
            return text, 0

        removed = 0
        out_lines: list[str] = []
        seen_lines: set[str] = set()

        for raw_line in text.split("\n"):
            line = raw_line
            canon = _canonical(line)

            if len(canon) >= self._min_len and canon in seen_lines:
                removed += 1
                continue
            if canon:
                seen_lines.add(canon)

            # Within the line, also dedupe repeated sentences.
            deduped_line, sent_removed = self._dedupe_sentences(line)
            removed += sent_removed
            out_lines.append(deduped_line)

        return "\n".join(out_lines), removed

    def _dedupe_sentences(self, line: str) -> tuple[str, int]:
        if "." not in line and "!" not in line and "?" not in line:
            return line, 0

        parts = _SENTENCE_SPLIT_RE.split(line)
        if len(parts) <= 1:
            return line, 0

        seen: set[str] = set()
        kept: list[str] = []
        removed = 0
        for p in parts:
            canon = _canonical(p)
            if len(canon) >= self._min_len and canon in seen:
                removed += 1
                continue
            if canon:
                seen.add(canon)
            kept.append(p)

        return " ".join(kept), removed
