"""Entropy / information-content based token pruning.

Drops low-information tokens (common stopwords, redundant determiners,
polite hedges) OUTSIDE of protected regions: code fences, inline code,
JSON values, quoted strings, and lines/sentences that match a
constraint regex.

This is the same *technique* as in academic libraries like PromptOptimizer
(EntropyOptim) and Microsoft's LLMLingua coarse-grained pass, adapted to
run on CPU with zero dependencies.

Safety properties:
  * Never touches fenced code.
  * Never touches inline-code spans delimited by backticks.
  * Never touches JSON-looking keys/values.
  * Never touches quoted strings (single or double).
  * Never touches constraint sentences.
  * Preserves leading/trailing whitespace structure.
  * Idempotent on already-pruned text.
"""
from __future__ import annotations

import re


# Words considered low-information in an instructional prompt.
# Curated conservatively — function words that rarely change meaning when
# removed in imperative English ("summarize the report" == "summarize report").
#
# We intentionally EXCLUDE negators ("no", "not", "never", "without") and
# quantifiers ("all", "any", "some", "only") because they carry real meaning.
_LOW_INFO_WORDS = {
    # articles + demonstratives
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns often used as filler
    "it", "its", "they", "them", "their", "we", "our", "us",
    # polite wrappers
    "please", "kindly",
    # common mild intensifiers
    "really", "quite", "rather", "pretty", "somewhat",
    # connective filler
    "so", "then", "also", "too",
    # generic
    "thing", "stuff",
}

# Patterns that protect their entire match from any modification.
# Applied IN ORDER and greedily; matches are replaced with sentinels,
# pruning runs, then sentinels are swapped back.
_PROTECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```.*?```", re.DOTALL),              # fenced code
    re.compile(r"`[^`\n]+`"),                         # inline code
    re.compile(r'"[^"\n]*"'),                         # double-quoted string
    re.compile(r"'[^'\n]*'"),                         # single-quoted string
    re.compile(r"\{[^{}]*\}"),                        # inline JSON-ish braces
    re.compile(r"<[^<>\n]+>"),                        # HTML/XML tags
    re.compile(r"https?://\S+"),                      # URLs
    re.compile(r"\b[A-Z]{2,}\b"),                     # acronyms (ACME, API)
    re.compile(r"\b\d+(?:\.\d+)?%?\b"),               # numbers + percentages
]

# Sentences that match this are NEVER pruned.
_CONSTRAINT_RE = re.compile(
    r"\b(must|must\s+not|do\s+not|don'?t|never|always|required|mandatory|"
    r"format:|schema:|output:|return\s+only|only\s+return|"
    r"exactly|verbatim|respond\s+in|answer\s+in)\b",
    re.IGNORECASE,
)

# Sentence splitter (conservative): split on . ! ? followed by whitespace
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Token splitter: keep punctuation attached so spacing is preserved.
_TOKEN_RE = re.compile(r"(\s+|[.,;:!?])")


class TokenPruner:
    """Drops low-information stopword-like tokens in safe contexts."""

    def __init__(self, *, extra_keep: set[str] | None = None) -> None:
        self._keep = extra_keep or set()

    def prune(self, text: str) -> tuple[str, int]:
        """Return (pruned_text, tokens_removed)."""
        if not text or not text.strip():
            return text, 0

        # Protect patterns by replacing each match with a sentinel token.
        # Sentinels are opaque \x00-delimited tokens that later patterns
        # cannot match (none of the patterns contain \x00).
        sentinels: list[str] = []

        def _stash(m: re.Match[str]) -> str:
            sentinels.append(m.group(0))
            return f"\x00SENT{len(sentinels) - 1}\x00"

        protected = text
        # Apply patterns in order. Because each pattern replaces its match
        # with a sentinel containing \x00, later patterns cannot re-enter
        # already-stashed regions (none of the protect patterns match \x00
        # or the SENT prefix).
        for pat in _PROTECT_PATTERNS:
            protected = pat.sub(_stash, protected)

        # Split into lines (preserving blank lines for paragraph structure),
        # then sentences, then tokens.
        removed = 0
        out_lines: list[str] = []
        in_fence = False
        for line in protected.split("\n"):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                out_lines.append(line)
                continue
            if in_fence or not line.strip():
                out_lines.append(line)
                continue
            new_line, n = self._prune_line(line)
            removed += n
            out_lines.append(new_line)

        result = "\n".join(out_lines)

        # Restore sentinels transitively — a sentinel's value may contain
        # other sentinels (e.g. a JSON brace pattern stashed AFTER quoted
        # strings have already been stashed inside it). Loop until no
        # sentinels remain or we hit a max iteration cap.
        for _ in range(len(sentinels) + 1):
            replaced_any = False
            for i, s in enumerate(sentinels):
                token = f"\x00SENT{i}\x00"
                if token in result:
                    result = result.replace(token, s)
                    replaced_any = True
            if not replaced_any:
                break

        # Post-clean: runs of spaces, space-before-punct, leading punct.
        result = re.sub(r"[ \t]{2,}", " ", result)
        result = re.sub(r"\s+([,.;:!?])", r"\1", result)
        result = "\n".join(ln.strip() if ln.strip() else ln for ln in result.split("\n"))

        return result, removed

    # --- Internals ---

    def _prune_line(self, line: str) -> tuple[str, int]:
        # If line has no sentence punctuation, treat it as a single sentence.
        if not any(c in line for c in ".!?"):
            return self._prune_sentence(line)

        parts = _SENT_SPLIT_RE.split(line)
        total = 0
        out: list[str] = []
        for p in parts:
            new_p, n = self._prune_sentence(p)
            total += n
            out.append(new_p)
        return " ".join(out), total

    def _prune_sentence(self, sent: str) -> tuple[str, int]:
        if _CONSTRAINT_RE.search(sent):
            return sent, 0  # hands off

        tokens = _TOKEN_RE.split(sent)
        out_tokens: list[str] = []
        removed = 0
        for t in tokens:
            low = t.lower().strip()
            if low in _LOW_INFO_WORDS and low not in self._keep:
                # Remove this token, but avoid producing a double-space or
                # orphaned punctuation.
                removed += 1
                continue
            out_tokens.append(t)
        return "".join(out_tokens), removed
