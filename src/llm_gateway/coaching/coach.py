"""Prompt Coach v2 — canonical markdown-structured rewrites.

The Coach's golden rule still holds:

    NEVER invent content the user did not provide.

What changed in v2:

  * Produces canonical markdown sections aligned with the OpenAI,
    Anthropic, and PromptBuilder 2025 best-practice frameworks:

        # Role              (only if the user supplied one — never fabricated)
        # Task              (the action-verb instructions)
        # Context           (background prose + delimited inputs)
        # Examples          (if any `e.g.`, `example:`, `input/output` pairs)
        # Format            (lifted format hints)
        # Constraints       (negations, must/must-not, boundaries)
        # Success Criteria  (measurable bits: word limits, acceptance bars)

  * Pasted data (fenced code blocks) is wrapped with ``<input>...</input>``
    tags — the delimiter both the OpenAI and Anthropic guides recommend
    as the most reliable for modern models.

  * Emits structured ``reasoning`` for every section it creates, plus
    ``suggestions`` for anything it refuses to fabricate (roles,
    examples, format when not stated, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .scorer import QualityReport


_TASK_VERB_RE = re.compile(
    r"\b("
    r"summarize|explain|write|generate|create|build|analyze|compare|"
    r"classify|translate|extract|list|describe|draft|review|refactor|"
    r"debug|solve|calculate|convert|identify|evaluate|rewrite|answer|"
    r"respond|design|implement|outline|critique|proofread|transform|"
    r"find|recommend|plan|propose|suggest|compose|define"
    r")\b",
    re.IGNORECASE,
)

_FORMAT_HINT_RE = re.compile(
    r"\b("
    r"in\s+json|as\s+bullet\s+points?|as\s+a\s+list|as\s+a\s+table|"
    r"in\s+markdown|as\s+markdown|return\s+only|answer\s+in|respond\s+in|"
    r"in\s+\d+\s+(?:words?|sentences?|bullets?|lines?|paragraphs?)|"
    r"exactly\s+\d+|no\s+more\s+than\s+\d+"
    r")\b",
    re.IGNORECASE,
)

_CONSTRAINT_SENT_RE = re.compile(
    r"\b(must\s+not|must|do\s+not|don'?t|never|always|avoid|exclude|only)\b",
    re.IGNORECASE,
)

_EXAMPLE_SENT_RE = re.compile(
    r"\b(for\s+example|e\.g\.|i\.e\.|example:|examples:|sample:)\b",
    re.IGNORECASE,
)

_SUCCESS_SENT_RE = re.compile(
    r"\b("
    r"under\s+\d+\s+(?:words?|lines?|tokens?)|"
    r"at\s+least\s+\d+|at\s+most\s+\d+|"
    r"no\s+more\s+than\s+\d+"
    r")\b",
    re.IGNORECASE,
)

_ALREADY_MD_RE = re.compile(
    r"^\s*#+\s*(role|task|context|examples?|format|output|constraints|success)\b",
    re.IGNORECASE | re.MULTILINE,
)
_OLD_STRUCTURED_RE = re.compile(
    r"^\s*(role|task|context|examples?|format|output|constraints|success\s+criteria)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

_LIKELY_INPUT_INTRO_RE = re.compile(
    r"\b(here\s+is|below\s+is|the\s+following|input:)\b",
    re.IGNORECASE,
)


@dataclass
class CoachingResult:
    rewritten_prompt: str
    applied: bool
    techniques_applied: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class PromptCoach:
    """Rewrites weak prompts into canonical markdown without inventing content."""

    def coach(
        self,
        prompt: str,
        quality: QualityReport,
        *,
        score_threshold: float,
    ) -> CoachingResult:
        # Above threshold or already structured → no-op but still hints.
        if (
            quality.score >= score_threshold
            or _ALREADY_MD_RE.search(prompt)
            or _OLD_STRUCTURED_RE.search(prompt)
        ):
            return CoachingResult(
                rewritten_prompt=prompt,
                applied=False,
                techniques_applied=[],
                reasoning=[],
                suggestions=self._hygiene_hints(quality),
            )

        rewritten, applied, reasoning = self._structure(prompt)

        # If structuring yielded nothing (too skeletal to restructure safely),
        # leave prompt and surface full suggestions.
        if not applied:
            return CoachingResult(
                rewritten_prompt=prompt,
                applied=False,
                techniques_applied=[],
                reasoning=[],
                suggestions=self._full_suggestions(quality),
            )

        return CoachingResult(
            rewritten_prompt=rewritten,
            applied=True,
            techniques_applied=applied,
            reasoning=reasoning,
            suggestions=self._non_applied_suggestions(quality, applied),
        )

    # --- Structuring pass ---

    def _structure(self, prompt: str) -> tuple[str, list[str], list[str]]:
        applied: list[str] = []
        reasoning: list[str] = []

        # Stash fenced code blocks — they must be preserved verbatim.
        fenced_blocks: list[str] = []
        working = prompt

        def _extract_fence(m: re.Match[str]) -> str:
            fenced_blocks.append(m.group(0))
            return f"\x00FENCE{len(fenced_blocks) - 1}\x00"

        working = re.sub(r"```.*?```", _extract_fence, working, flags=re.DOTALL)

        # Sentence split (after fences are stashed, so we don't split inside code)
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", working.strip())
            if s.strip()
        ]
        if not sentences:
            sentences = [working.strip()]

        task_sents: list[str] = []
        context_sents: list[str] = []
        constraint_sents: list[str] = []
        format_sents: list[str] = []
        example_sents: list[str] = []
        success_sents: list[str] = []
        introducer_triggered = False

        for s in sentences:
            if _LIKELY_INPUT_INTRO_RE.search(s):
                context_sents.append(s)
                introducer_triggered = True
                continue
            if _SUCCESS_SENT_RE.search(s):
                success_sents.append(s)
                continue
            if _FORMAT_HINT_RE.search(s):
                format_sents.append(s)
                continue
            if _CONSTRAINT_SENT_RE.search(s):
                constraint_sents.append(s)
                continue
            if _EXAMPLE_SENT_RE.search(s):
                example_sents.append(s)
                continue
            if _TASK_VERB_RE.search(s) and not task_sents:
                task_sents.append(s)
                continue
            if task_sents or introducer_triggered:
                context_sents.append(s)
            else:
                task_sents.append(s)

        # Need a task OR a fenced example to justify restructuring.
        if not task_sents and not fenced_blocks:
            return prompt, [], []

        sections: list[str] = []

        if task_sents:
            sections.append("# Task\n" + " ".join(task_sents))
            applied.append("structure:task")
            reasoning.append(
                "Front-loaded the action-verb instruction into a `# Task` "
                "section — models attend most reliably when the ask appears "
                "first."
            )

        if context_sents or fenced_blocks:
            ctx_parts: list[str] = []
            if context_sents:
                ctx_parts.append(" ".join(context_sents))
            # Wrap each pasted fenced block in <input> tags (OpenAI/Anthropic
            # recommended delimiter for inputs).
            for i, block in enumerate(fenced_blocks):
                ctx_parts.append(f"<input>\n{block}\n</input>")
            sections.append("# Context\n" + "\n\n".join(ctx_parts))
            applied.append("structure:context")
            reasoning.append(
                "Grouped background material under `# Context` and wrapped "
                "pasted inputs in `<input>` tags so the model can "
                "distinguish data from instructions."
            )

        if example_sents:
            sections.append("# Examples\n" + " ".join(example_sents))
            applied.append("structure:examples")
            reasoning.append(
                "Lifted example/demonstration sentences into a dedicated "
                "`# Examples` section."
            )

        if format_sents:
            sections.append("# Format\n" + " ".join(format_sents))
            applied.append("structure:format")
            reasoning.append(
                "Surfaced output-format hints into an explicit `# Format` "
                "section so the model cannot overlook them."
            )

        if constraint_sents:
            bullets = "\n".join(f"- {s}" for s in constraint_sents)
            sections.append("# Constraints\n" + bullets)
            applied.append("structure:constraints")
            reasoning.append(
                "Collected negations (do-not / never / must-not / only) "
                "under a `# Constraints` section."
            )

        if success_sents:
            sections.append("# Success Criteria\n" + " ".join(success_sents))
            applied.append("structure:success_criteria")
            reasoning.append(
                "Extracted measurable acceptance criteria into a "
                "`# Success Criteria` section."
            )

        rewritten = "\n\n".join(sections)
        # Restore any fences that the structuring didn't place explicitly.
        for i, block in enumerate(fenced_blocks):
            rewritten = rewritten.replace(f"\x00FENCE{i}\x00", block)

        return rewritten, applied, reasoning

    # --- Hint surfaces ---

    def _hygiene_hints(self, quality: QualityReport) -> list[str]:
        out: list[str] = []
        if quality.score < 0.85:
            for d in quality.dimensions:
                if d.score < 0.5:
                    out.append(f"{d.name}: {d.reason}")
        return out

    def _full_suggestions(self, quality: QualityReport) -> list[str]:
        suggestions: list[str] = []
        missing = set(quality.techniques_missing())

        if "role_persona" in missing:
            suggestions.append(
                "Add a role/persona (e.g. 'You are an experienced technical "
                "writer'). The coach never adds one on your behalf."
            )
        if "clear_action_verb" in missing:
            suggestions.append(
                "State the action explicitly ('summarize', 'list', "
                "'generate'…). Without a verb, the model has to guess."
            )
        if "output_format" in missing:
            suggestions.append(
                "Specify the output format: 'as JSON', 'as 3 bullet points', "
                "'under 200 words', 'as a markdown table'."
            )
        if "provide_context" in missing:
            suggestions.append(
                "Provide context or the input the model should work on. "
                "Delimit pasted material with triple backticks or <input> tags."
            )
        if "state_constraints" in missing:
            suggestions.append(
                "List explicit constraints — what to include, exclude, or "
                "avoid ('Do not mention competitors', 'only return JSON')."
            )
        if "few_shot_examples" in missing:
            suggestions.append(
                "Show 1–3 examples (few-shot). Demonstrations dramatically "
                "improve consistency for structured or classification tasks."
            )
        if "add_measurable_criteria" in missing:
            suggestions.append(
                "Add measurable success criteria — numbers, explicit "
                "examples, or a precise definition of done."
            )
        return suggestions

    def _non_applied_suggestions(
        self, quality: QualityReport, applied: list[str]
    ) -> list[str]:
        suggestions: list[str] = []
        missing = set(quality.techniques_missing())

        if "role_persona" in missing:
            suggestions.append(
                "Consider adding a `# Role` section (e.g. 'You are an "
                "experienced technical writer'). The coach never invents "
                "a role on your behalf."
            )
        if "few_shot_examples" in missing and "structure:examples" not in applied:
            suggestions.append(
                "Consider adding 1–3 input→output examples. Few-shot "
                "demonstrations dramatically improve consistency."
            )
        if "output_format" in missing and "structure:format" not in applied:
            suggestions.append(
                "Specify an output format in a `# Format` section "
                "(JSON / bullets / word limit / schema)."
            )
        if (
            "add_measurable_criteria" in missing
            and "structure:success_criteria" not in applied
        ):
            suggestions.append(
                "Add measurable success criteria (word limits, required "
                "fields, explicit acceptance bar)."
            )
        return suggestions
