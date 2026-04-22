"""Prompt quality scoring.

Scores a prompt against six prompt-engineering dimensions, each producing
a 0.0-1.0 sub-score. The composite is a weighted average.

Dimensions:
  role        — does the prompt establish who the model should be?
  task        — is there a clear, actionable instruction?
  context     — is relevant background / input provided?
  format      — is the desired output shape specified?
  constraints — are rules (do/don't, must/must not) stated?
  specificity — measurable criteria, concrete definitions of success?

The scorer is deliberately heuristic and deterministic. It is not a
substitute for human judgment; it is a signal used by the Coach to
decide whether to rewrite and what scaffolding to add.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# --- Regex signals ------------------------------------------------------

_ROLE_PATTERNS = [
    re.compile(r"\byou\s+are\s+(?:a|an|the)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:a|an|the)\b", re.IGNORECASE),
    re.compile(r"\bas\s+(?:a|an)\s+\w+,", re.IGNORECASE),
    re.compile(r"\bassume\s+the\s+role\s+of\b", re.IGNORECASE),
    re.compile(r"\brole:\s*\w+", re.IGNORECASE),
    re.compile(r"\bpretend\s+(?:to\s+be|you\s+are)\b", re.IGNORECASE),
]

_TASK_VERBS = {
    "summarize", "explain", "write", "generate", "create", "build",
    "analyze", "compare", "classify", "translate", "extract", "list",
    "describe", "draft", "review", "refactor", "debug", "solve",
    "calculate", "convert", "identify", "evaluate", "rewrite",
    "answer", "respond", "design", "implement", "outline",
    "critique", "proofread", "transform", "find", "recommend",
    "plan", "propose", "suggest", "compose", "define",
}

_VAGUE_PHRASES = [
    re.compile(r"\bhelp\s+me\s+with\s+this\b", re.IGNORECASE),
    re.compile(r"\bcan\s+you\s+do\s+(this|that|it)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+do\s+you\s+think\b", re.IGNORECASE),
    re.compile(r"\bjust\s+(do|make)\s+(it|something)\b", re.IGNORECASE),
    re.compile(r"^\s*(hi|hello|hey)\b", re.IGNORECASE),
]

_CONTEXT_SIGNALS = [
    re.compile(r"\bgiven\b", re.IGNORECASE),
    re.compile(r"\bcontext:\b", re.IGNORECASE),
    re.compile(r"\bbackground:\b", re.IGNORECASE),
    re.compile(r"\bhere\s+is\b", re.IGNORECASE),
    re.compile(r"\bbelow\s+is\b", re.IGNORECASE),
    re.compile(r"\bthe\s+following\b", re.IGNORECASE),
    re.compile(r"\binput:\b", re.IGNORECASE),
    re.compile(r"```"),  # fenced blocks almost always carry context
    re.compile(r"\buse\s+the\s+(?:data|text|code|information)\s+below\b", re.IGNORECASE),
]

_FORMAT_SIGNALS = [
    re.compile(r"\bin\s+json\b", re.IGNORECASE),
    re.compile(r"\bas\s+(?:a\s+)?(?:bullet\s+points?|list|table|markdown)\b", re.IGNORECASE),
    re.compile(r"\bformat:\b", re.IGNORECASE),
    re.compile(r"\boutput:\b", re.IGNORECASE),
    re.compile(r"\bschema:\b", re.IGNORECASE),
    re.compile(r"\breturn\s+(?:a|an|only)\b", re.IGNORECASE),
    re.compile(r"\brespond\s+(?:with|in)\b", re.IGNORECASE),
    re.compile(r"\banswer\s+in\b", re.IGNORECASE),
    re.compile(r"\b(?:in|with)\s+\d+\s+(?:words?|sentences?|bullets?|lines?|paragraphs?)\b", re.IGNORECASE),
    re.compile(r"\bexactly\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bno\s+more\s+than\s+\d+\b", re.IGNORECASE),
]

_CONSTRAINT_SIGNALS = [
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b|\bdon'?t\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\brequired\b", re.IGNORECASE),
    re.compile(r"\bonly\b", re.IGNORECASE),
    re.compile(r"\bavoid\b", re.IGNORECASE),
    re.compile(r"\bexclude\b", re.IGNORECASE),
]

# A number, or a word that establishes a concrete criterion.
_SPECIFICITY_SIGNALS = [
    re.compile(r"\b\d+\b"),  # any number
    re.compile(r"\b(?:specifically|precisely|exactly|concretely)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+example\b", re.IGNORECASE),
    re.compile(r"\be\.g\.\b", re.IGNORECASE),
    re.compile(r"\bi\.e\.\b", re.IGNORECASE),
]


# --- Weights ------------------------------------------------------------
# Task is weighted highest because a prompt without a clear task is
# fundamentally broken. Role is lowest because not every prompt needs one.

_WEIGHTS: dict[str, float] = {
    "task":        0.25,
    "format":      0.18,
    "context":     0.14,
    "specificity": 0.13,
    "constraints": 0.10,
    "examples":    0.10,
    "role":        0.10,
}


# Examples signals: in-prompt demonstrations
_EXAMPLE_SIGNALS = [
    re.compile(r"\bfor\s+example\b", re.IGNORECASE),
    re.compile(r"\be\.g\.\b", re.IGNORECASE),
    re.compile(r"\bi\.e\.\b", re.IGNORECASE),
    re.compile(r"\bexample:\b", re.IGNORECASE),
    re.compile(r"\bexamples:\b", re.IGNORECASE),
    re.compile(r"\bsample:\b", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:input|output):\s", re.IGNORECASE),  # few-shot style
    re.compile(r"->"),  # "Negative -> 0" style
    re.compile(r"```"),  # code blocks often ARE examples
]


@dataclass
class DimensionScore:
    name: str
    score: float                 # 0.0 - 1.0
    weight: float
    reason: str                  # human-readable explanation
    techniques_missing: list[str]  # technique hints this dimension would unlock


@dataclass
class QualityReport:
    score: float                 # composite 0.0 - 1.0
    dimensions: list[DimensionScore]
    strengths: list[str]
    weaknesses: list[str]

    def techniques_missing(self) -> list[str]:
        """Flattened unique list of techniques suggested by weak dimensions."""
        seen: set[str] = set()
        out: list[str] = []
        for d in self.dimensions:
            for t in d.techniques_missing:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out


class PromptQualityScorer:
    """Deterministic rubric-based scorer. No external calls."""

    # Prompts shorter than this are treated as under-developed regardless
    # of content — you cannot say much in four words.
    MIN_MEANINGFUL_WORDS = 4

    def score(self, prompt: str) -> QualityReport:
        words = prompt.split()
        word_count = len(words)

        dims: list[DimensionScore] = []
        dims.append(self._score_role(prompt))
        dims.append(self._score_task(prompt, word_count))
        dims.append(self._score_context(prompt, word_count))
        dims.append(self._score_format(prompt))
        dims.append(self._score_constraints(prompt))
        dims.append(self._score_examples(prompt))
        dims.append(self._score_specificity(prompt, word_count))

        # Composite
        composite = sum(d.score * d.weight for d in dims)

        # Very short prompts get a global multiplier penalty — a four-word
        # prompt cannot be "high quality" no matter how well scored.
        if word_count < self.MIN_MEANINGFUL_WORDS:
            composite *= 0.4

        strengths = [d.reason for d in dims if d.score >= 0.7]
        weaknesses = [d.reason for d in dims if d.score < 0.5]

        return QualityReport(
            score=round(max(0.0, min(1.0, composite)), 4),
            dimensions=dims,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    # --- Per-dimension ---

    def _score_role(self, prompt: str) -> DimensionScore:
        if any(p.search(prompt) for p in _ROLE_PATTERNS):
            return DimensionScore(
                "role", 1.0, _WEIGHTS["role"],
                "Role/persona is established.",
                [],
            )
        return DimensionScore(
            "role", 0.0, _WEIGHTS["role"],
            "No role or persona specified.",
            ["role_persona"],
        )

    def _score_task(self, prompt: str, word_count: int) -> DimensionScore:
        lower_words = {w.strip(".,!?;:\"'()").lower() for w in prompt.split()}
        has_verb = bool(lower_words & _TASK_VERBS)
        is_vague = any(p.search(prompt) for p in _VAGUE_PHRASES)

        if is_vague and word_count < 10:
            return DimensionScore(
                "task", 0.1, _WEIGHTS["task"],
                "Task is vague ('help me', 'do this', etc.) without specifics.",
                ["clear_action_verb", "concrete_object"],
            )
        if has_verb and not is_vague:
            return DimensionScore(
                "task", 1.0, _WEIGHTS["task"],
                "Clear task with action verb.",
                [],
            )
        if has_verb and is_vague:
            return DimensionScore(
                "task", 0.5, _WEIGHTS["task"],
                "Task verb present but vague phrasing reduces clarity.",
                ["remove_vague_phrases"],
            )
        return DimensionScore(
            "task", 0.3, _WEIGHTS["task"],
            "No clear action verb found.",
            ["clear_action_verb"],
        )

    def _score_context(self, prompt: str, word_count: int) -> DimensionScore:
        has_signal = any(p.search(prompt) for p in _CONTEXT_SIGNALS)
        if has_signal:
            return DimensionScore(
                "context", 1.0, _WEIGHTS["context"],
                "Context or input is provided.",
                [],
            )
        # For prompts under ~15 words, lack of context is often fine
        # (it's a quick question). For longer prompts without any context
        # markers, that's a real gap.
        if word_count < 15:
            return DimensionScore(
                "context", 0.6, _WEIGHTS["context"],
                "No explicit context, but the prompt is short enough that "
                "it may not be needed.",
                [],
            )
        return DimensionScore(
            "context", 0.2, _WEIGHTS["context"],
            "Long prompt without explicit context or inputs.",
            ["provide_context", "delimit_inputs"],
        )

    def _score_format(self, prompt: str) -> DimensionScore:
        if any(p.search(prompt) for p in _FORMAT_SIGNALS):
            return DimensionScore(
                "format", 1.0, _WEIGHTS["format"],
                "Output format is specified.",
                [],
            )
        return DimensionScore(
            "format", 0.0, _WEIGHTS["format"],
            "Output format is not specified.",
            ["output_format"],
        )

    def _score_constraints(self, prompt: str) -> DimensionScore:
        hits = sum(1 for p in _CONSTRAINT_SIGNALS if p.search(prompt))
        if hits >= 2:
            return DimensionScore(
                "constraints", 1.0, _WEIGHTS["constraints"],
                "Multiple constraints are stated.",
                [],
            )
        if hits == 1:
            return DimensionScore(
                "constraints", 0.6, _WEIGHTS["constraints"],
                "Some constraints are stated.",
                [],
            )
        return DimensionScore(
            "constraints", 0.2, _WEIGHTS["constraints"],
            "No explicit constraints (do/don't, must/avoid).",
            ["state_constraints"],
        )

    def _score_examples(self, prompt: str) -> DimensionScore:
        hits = sum(1 for p in _EXAMPLE_SIGNALS if p.search(prompt))
        if hits >= 2:
            return DimensionScore(
                "examples", 1.0, _WEIGHTS["examples"],
                "Prompt includes examples / demonstrations.",
                [],
            )
        if hits == 1:
            return DimensionScore(
                "examples", 0.5, _WEIGHTS["examples"],
                "Some example usage is present; adding a second "
                "demonstration typically improves consistency.",
                ["few_shot_examples"],
            )
        return DimensionScore(
            "examples", 0.15, _WEIGHTS["examples"],
            "No examples provided. Few-shot examples dramatically improve "
            "output consistency for structured tasks.",
            ["few_shot_examples"],
        )

    def _score_specificity(self, prompt: str, word_count: int) -> DimensionScore:
        hits = sum(1 for p in _SPECIFICITY_SIGNALS if p.search(prompt))
        if hits >= 2:
            return DimensionScore(
                "specificity", 1.0, _WEIGHTS["specificity"],
                "Prompt includes concrete, measurable criteria.",
                [],
            )
        if hits == 1 and word_count >= 8:
            return DimensionScore(
                "specificity", 0.6, _WEIGHTS["specificity"],
                "Some specificity; could be sharper.",
                ["add_measurable_criteria"],
            )
        return DimensionScore(
            "specificity", 0.2, _WEIGHTS["specificity"],
            "No measurable criteria or concrete success definition.",
            ["add_measurable_criteria"],
        )
