"""Attribute extraction for the mirror engine.

The whole point of a mirror is that the human and AI documents match on topic, genre,
length and structure, so the only systematic difference left is the generation process.
A detector trained on "write an article about X" versus real scraped web text mostly
learns domain, and it collapses the moment the domain shifts.

The hard constraint here is that extraction must never copy sentences from the source.
If it did, the generated mirror would share surface text with its human counterpart and
the classifier would learn copy detection instead of an authorship signal. That is not
a style preference, it is the difference between measuring what we claim to measure and
measuring nothing. `assert_no_verbatim_copy` enforces it and the runner calls it on
every extraction.

Two extractors:
  HeuristicExtractor - no model, no network, deterministic. Good enough for structure,
      length and difficulty; crude on topic. This is what runs in tests and what makes
      the pipeline verifiable offline.
  LLMExtractor - a small local instruct model doing the same job better. Phase 2
      deployment. Its output goes through the same no-copy guard.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Protocol

_WORD = re.compile(r"\b[a-zA-Z][a-zA-Z'-]+\b")
_SENT = re.compile(r"[.!?]+(?:\s|$)")

# Deliberately small and boring. A real stoplist belongs in a data file; this exists so
# topic extraction does not return "the".
_STOP = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as", "was",
    "on", "be", "by", "this", "are", "from", "at", "or", "an", "have", "not", "but",
    "had", "has", "were", "which", "their", "they", "its", "been", "would", "there",
    "when", "into", "than", "then", "them", "these", "those", "such", "some", "other",
    "more", "most", "also", "over", "after", "before", "while", "during", "each",
    "about", "through", "between", "under", "against", "because", "however", "several",
    "being", "could", "should", "will", "may", "must", "very", "many", "much", "own",
}

MAX_ANCHOR_WORDS = 4
VERBATIM_SPAN_LIMIT = 8  # an anchor may not reproduce this many consecutive source words


@dataclass
class MirrorAttributes:
    topic: str
    genre: str
    register: str
    target_tokens: int
    structure: str
    difficulty: str
    key_anchors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def prompt_fields(self) -> dict[str, str]:
        d = self.as_dict()
        d["key_anchors"] = "\n".join(f"- {a}" for a in self.key_anchors)
        d["target_tokens"] = str(self.target_tokens)
        return d


class VerbatimCopyError(ValueError):
    """Raised when an extracted attribute reproduces source text. See module docstring."""


def _word_list(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def assert_no_verbatim_copy(attrs: MirrorAttributes, source_text: str) -> None:
    """Fail if any anchor is a long verbatim span of the source, or is over-long.

    Short overlaps are unavoidable and harmless: a document about soil chemistry will
    have "soil chemistry" in both. What must not survive is a reproduced clause.
    """
    src = _word_list(source_text)
    src_spans = {
        " ".join(src[i : i + VERBATIM_SPAN_LIMIT]) for i in range(max(len(src) - VERBATIM_SPAN_LIMIT + 1, 0))
    }
    for anchor in attrs.key_anchors:
        words = _word_list(anchor)
        if len(words) > MAX_ANCHOR_WORDS:
            raise VerbatimCopyError(
                f"anchor {anchor!r} has {len(words)} words, limit is {MAX_ANCHOR_WORDS}; "
                "anchors are keyphrases, not sentences"
            )
        for i in range(max(len(words) - VERBATIM_SPAN_LIMIT + 1, 0)):
            if " ".join(words[i : i + VERBATIM_SPAN_LIMIT]) in src_spans:
                raise VerbatimCopyError(f"anchor {anchor!r} reproduces source text verbatim")


class Extractor(Protocol):
    name: str

    def extract(self, text: str) -> MirrorAttributes: ...


def detect_structure(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "prose"
    bulleted = sum(1 for ln in lines if re.match(r"^([-*•]|\d+[.)])\s", ln))
    if bulleted / len(lines) > 0.3:
        return "listy"
    quoted = sum(1 for ln in lines if ln.startswith(('"', "'", "-")) and len(ln) < 120)
    if quoted / len(lines) > 0.4:
        return "dialogue"
    short_lines = sum(1 for ln in lines if len(ln) < 60)
    if len(lines) > 4 and short_lines / len(lines) > 0.6:
        return "sectioned"
    return "prose"


def detect_difficulty(text: str) -> str:
    words = _word_list(text)
    if not words:
        return "moderate"
    sentences = max(len(_SENT.findall(text)), 1)
    words_per_sentence = len(words) / sentences
    long_word_share = sum(1 for w in words if len(w) >= 8) / len(words)
    # Two crude proxies for the two things readability formulas actually measure.
    score = words_per_sentence / 20.0 + long_word_share / 0.15
    if score < 1.0:
        return "plain"
    if score < 1.8:
        return "moderate"
    return "demanding"


def detect_genre(text: str, structure: str) -> str:
    lowered = text.lower()
    if structure == "listy":
        return "list-based guide"
    if structure == "dialogue":
        return "dialogue"
    if any(k in lowered for k in ("we find", "this paper", "the results", "we show", "abstract")):
        return "research-style summary"
    if any(k in lowered for k in ("i remember", "my father", "she said", "he said")):
        return "narrative piece"
    if any(k in lowered for k in ("shall", "pursuant", "hereby", "the committee", "section")):
        return "formal report"
    return "explainer"


def detect_register(text: str, difficulty: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("you'll", "let's", "we've", "i think", "you can")):
        return "conversational"
    if difficulty == "demanding":
        return "formal"
    return "informational"


class HeuristicExtractor:
    """Deterministic, dependency-free. Runs in tests and makes Phase 2 verifiable offline."""

    name = "heuristic_v1"

    def __init__(self, anchors_min: int = 3, anchors_max: int = 6) -> None:
        self.anchors_min, self.anchors_max = anchors_min, anchors_max

    def extract(self, text: str) -> MirrorAttributes:
        words = [w for w in _word_list(text) if w not in _STOP and len(w) > 3]
        freq = Counter(words)
        top = [w for w, _ in freq.most_common(self.anchors_max)]
        topic = ", ".join(top[:3]) if top else "an unspecified subject"
        structure = detect_structure(text)
        difficulty = detect_difficulty(text)
        attrs = MirrorAttributes(
            topic=topic,
            genre=detect_genre(text, structure),
            register=detect_register(text, difficulty),
            target_tokens=len(text.split()),
            structure=structure,
            difficulty=difficulty,
            # Single words. Cannot reproduce a clause by construction.
            key_anchors=top[: self.anchors_max] or ["the subject of the document"],
        )
        assert_no_verbatim_copy(attrs, text)
        return attrs


class LLMExtractor:
    """Phase 2 deployment. A small instruct model produces the same attribute set.

    Its output is untrusted and goes through assert_no_verbatim_copy like everything
    else: an LLM asked to summarize will happily quote a sentence back.
    """

    name = "llm_v1"

    def __init__(self, model_id: str, revision: str) -> None:
        self.model_id, self.revision = model_id, revision

    def extract(self, text: str) -> MirrorAttributes:
        raise NotImplementedError("Phase 2 deployment: wire a local instruct model")
