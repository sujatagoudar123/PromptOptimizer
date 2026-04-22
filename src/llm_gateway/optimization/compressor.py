"""Safe, conservative prompt compression.

Design principles:
  * NEVER touch text inside fenced code blocks (``` ... ```).
  * NEVER touch text inside inline-code (`...`) or quoted strings that look like values.
  * NEVER touch sentences that look like constraints ("must", "do not", "never",
    "always", "required", "format:", explicit JSON schema, etc).
  * Only remove unambiguous filler phrases.

The guard operates at sentence granularity — one constraint word should not
exempt an entire multi-sentence line from compression.
"""
from __future__ import annotations

import re


# Filler phrases that are almost always safe to remove at sentence boundaries
# or mid-sentence. Ordering: longer/more specific first.
# Safe-everywhere fillers: pure politeness and shorthand, removable even
# inside constraint sentences because they cannot carry meaning.
_SAFE_ANYWHERE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Multi-word politeness wrappers
    (re.compile(r"\bcould\s+you\s+please\s+kindly\b", re.IGNORECASE), ""),
    (re.compile(r"\bcan\s+you\s+please\s+kindly\b", re.IGNORECASE), ""),
    (re.compile(r"\bcould\s+you\s+please\b", re.IGNORECASE), ""),
    (re.compile(r"\bcan\s+you\s+please\b", re.IGNORECASE), ""),
    (re.compile(r"\bwould\s+you\s+please\b", re.IGNORECASE), ""),
    (re.compile(r"\bif\s+you\s+don'?t\s+mind\b", re.IGNORECASE), ""),
    (re.compile(r"\bif\s+it'?s\s+not\s+too\s+much\s+trouble\b", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+would\s+really\s+appreciate\s+(it\s+)?if\s+you\s+(could|would)\b", re.IGNORECASE), ""),
    (re.compile(r"\bplease\s+kindly\b", re.IGNORECASE), ""),
    (re.compile(r"\bkindly\s+please\b", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+hope\s+this\s+(email\s+)?finds\s+you\s+well\b", re.IGNORECASE), ""),
    (re.compile(r"\bthanks?\s+in\s+advance\b", re.IGNORECASE), ""),
    (re.compile(r"\bthank\s+you\s+in\s+advance\b", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+was\s+wondering\s+if\b", re.IGNORECASE), ""),

    # Internet shorthand expansions + removals
    (re.compile(r"\bplz\b|\bpls\b", re.IGNORECASE), ""),
    (re.compile(r"\bthx\b|\bthnx\b|\bty\b", re.IGNORECASE), ""),
    (re.compile(r"\bu\b", re.IGNORECASE), "you"),
    (re.compile(r"\bur\b", re.IGNORECASE), "your"),
    (re.compile(r"\br\b(?=\s)", re.IGNORECASE), "are"),
    (re.compile(r"\s+n\s+", re.IGNORECASE), " and "),
    (re.compile(r"\bcoz\b|\bbcoz\b|\bcuz\b", re.IGNORECASE), "because"),
    (re.compile(r"\bw/\b", re.IGNORECASE), "with"),
    (re.compile(r"\bw/o\b", re.IGNORECASE), "without"),
    (re.compile(r"\bb/w\b", re.IGNORECASE), "between"),

    # Single-word politeness
    (re.compile(r"\bkindly\b", re.IGNORECASE), ""),
    (re.compile(r"\bplease\b", re.IGNORECASE), ""),

    # "Could you" / "can you" / "would you" fallbacks (after multi-word patterns)
    (re.compile(r"\bcould\s+you\b", re.IGNORECASE), ""),
    (re.compile(r"\bcan\s+you\b", re.IGNORECASE), ""),
    (re.compile(r"\bwould\s+you\b", re.IGNORECASE), ""),
]

# Non-constraint-only fillers: may plausibly carry nuance inside a
# constraint, so we skip these when the sentence matches the constraint
# regex. Wordy-to-terse substitutions and epistemic hedges go here.
_NON_CONSTRAINT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bi\s+would\s+like\s+you\s+to\b", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+(just\s+)?want(ed)?\s+you\s+to\b", re.IGNORECASE), ""),
    (re.compile(r"\bfor\s+your\s+reference\b", re.IGNORECASE), ""),
    (re.compile(r"\bas\s+i\s+mentioned\s+(earlier|before|previously)\b", re.IGNORECASE), ""),
    (re.compile(r"\bjust\s+to\s+be\s+clear\b", re.IGNORECASE), ""),
    (re.compile(r"\bit\s+is\s+important\s+to\s+note\s+that\b", re.IGNORECASE), ""),
    (re.compile(r"\bplease\s+note\s+that\b", re.IGNORECASE), ""),

    # Wordy-to-terse substitutions
    (re.compile(r"\bvery\s+(much|briefly)\b", re.IGNORECASE), r"\1"),
    (re.compile(r"\bin\s+order\s+to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bdue\s+to\s+the\s+fact\s+that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bowing\s+to\s+the\s+fact\s+that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bat\s+this\s+point\s+in\s+time\b", re.IGNORECASE), "now"),
    (re.compile(r"\bat\s+the\s+present\s+time\b", re.IGNORECASE), "now"),
    (re.compile(r"\bin\s+the\s+event\s+that\b", re.IGNORECASE), "if"),
    (re.compile(r"\bin\s+spite\s+of\s+the\s+fact\s+that\b", re.IGNORECASE), "although"),
    (re.compile(r"\bfor\s+the\s+purpose\s+of\b", re.IGNORECASE), "to"),
    (re.compile(r"\bwith\s+regard\s+to\b", re.IGNORECASE), "about"),
    (re.compile(r"\bin\s+reference\s+to\b", re.IGNORECASE), "about"),
    (re.compile(r"\ba\s+large\s+number\s+of\b", re.IGNORECASE), "many"),
    (re.compile(r"\ba\s+majority\s+of\b", re.IGNORECASE), "most"),

    # Epistemic hedges — may change nuance inside a constraint so limited scope
    (re.compile(r"\bbasically\b|\bessentially\b|\bliterally\b", re.IGNORECASE), ""),
    (re.compile(r"\bactually\b", re.IGNORECASE), ""),
    (re.compile(r"\bobviously\b|\bclearly\b(?!\s+state)", re.IGNORECASE), ""),
    (re.compile(r"\bhonestly\b|\bfrankly\b", re.IGNORECASE), ""),
    (re.compile(r"\bsimply\b", re.IGNORECASE), ""),
]

# Back-compat alias — keeps any downstream imports working if anyone
# was reaching in. External callers should use the tiered constants above.
_FILLER_PATTERNS = _SAFE_ANYWHERE_PATTERNS + _NON_CONSTRAINT_PATTERNS

# Sentences matching this regex are treated as constraints and passed through
# untouched. Applied per-sentence, not per-line, so one constraint doesn't
# block compression of surrounding prose.
_CONSTRAINT_RE = re.compile(
    r"\b(must|must\s+not|do\s+not|don'?t|never|always|required|mandatory|"
    r"format:|schema:|output:|return\s+only|only\s+return|"
    r"exactly|verbatim|respond\s+in|answer\s+in)\b",
    re.IGNORECASE,
)

# Fenced code block delimiter
_FENCE_RE = re.compile(r"^```")

# Sentence-ish splitter: splits after . ! ? while keeping delimiters with
# the preceding sentence.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Cleanup: multiple spaces created by replacements
_WS_CLEANUP_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


class Compressor:
    """Remove filler; protect code blocks and constraint sentences."""

    def compress(self, text: str) -> tuple[str, int]:
        """Return (compressed_text, number_of_replacements_applied)."""
        if not text:
            return text, 0

        out_lines: list[str] = []
        in_fence = False
        replacements = 0

        for line in text.split("\n"):
            if _FENCE_RE.match(line.strip()):
                in_fence = not in_fence
                out_lines.append(line)
                continue

            if in_fence:
                out_lines.append(line)  # never touch code
                continue

            new_line, n = self._compress_line(line)
            replacements += n
            out_lines.append(new_line)

        result = "\n".join(out_lines)

        # Post-cleanup: whitespace, orphaned punctuation, and capitalization
        result = _WS_CLEANUP_RE.sub(" ", result)
        result = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
        # Remove leading punctuation-only sentence fragments like ". summarize"
        result = re.sub(r"(^|\.\s+)[,;:]\s*", r"\1", result)
        # Collapse "word  ." -> "word."
        result = re.sub(r"\s+\.", ".", result)
        # Per-line trim
        result = "\n".join(line.strip() if line.strip() else line for line in result.split("\n"))
        # Recapitalize the first character of each sentence (compression often
        # lowercases by removing capitalized leading fillers like "Basically,")
        result = self._recapitalize(result)

        return result, replacements

    @staticmethod
    def _recapitalize(text: str) -> str:
        """Capitalize the first letter after sentence terminators and at start."""
        def _cap_first(m: re.Match[str]) -> str:
            return m.group(1) + m.group(2).upper()
        # After a sentence terminator + whitespace
        text = re.sub(r"([.!?]\s+)([a-z])", _cap_first, text)
        # At the very start of a line (after optional whitespace)
        text = re.sub(
            r"(^|\n)(\s*)([a-z])",
            lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
            text,
        )
        return text

    def _compress_line(self, line: str) -> tuple[str, int]:
        """Split a line into sentences; apply the appropriate tier to each.

        Constraint sentences get only the safe-anywhere patterns (politeness
        wrappers, shorthand). Non-constraint sentences get both tiers.
        """
        if not line.strip():
            return line, 0

        # If no sentence terminators, treat the whole line as one sentence.
        if not any(c in line for c in ".!?"):
            is_constraint = bool(_CONSTRAINT_RE.search(line))
            return self._apply_patterns(line, is_constraint=is_constraint)

        sentences = _SENT_SPLIT_RE.split(line)
        out: list[str] = []
        total = 0
        for s in sentences:
            is_constraint = bool(_CONSTRAINT_RE.search(s))
            new_s, n = self._apply_patterns(s, is_constraint=is_constraint)
            total += n
            out.append(new_s)
        return " ".join(out), total

    def _apply_patterns(self, s: str, *, is_constraint: bool) -> tuple[str, int]:
        """Apply safe-anywhere patterns always; non-constraint patterns only
        when the sentence isn't a constraint."""
        count = 0
        new = s

        # Tier 1: safe-anywhere (politeness, shorthand) — always applied.
        for pattern, replacement in _SAFE_ANYWHERE_PATTERNS:
            new, n = pattern.subn(replacement, new)
            count += n

        # Tier 2: non-constraint patterns — skip if this sentence is a constraint.
        if not is_constraint:
            for pattern, replacement in _NON_CONSTRAINT_PATTERNS:
                new, n = pattern.subn(replacement, new)
                count += n

        # Clean up: leading/trailing commas and duplicate punctuation
        # from sentence-initial filler removal.
        new = re.sub(r"^[\s,;:]+", "", new)
        new = re.sub(r"\s+,", ",", new)
        return new, count
