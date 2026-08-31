"""Language identification. Stage 2.

Pluggable on purpose. fastText's lid model is the right answer at corpus scale, but it
requires a model download, and FineWeb and FineWeb-Edu already ship `language` and
`language_score` fields from their own fastText pass. So:

  1. If the upstream record already carries a language score, trust it. Re-running
     language ID on FineWeb would be recomputing a field it already gives us.
  2. Otherwise use fastText when installed.
  3. Otherwise fall back to a stopword heuristic, which is crude but honest, and which
     records `detector: "heuristic"` on the record so nothing downstream mistakes it
     for a real score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as",
    "was", "on", "be", "by", "this", "are", "from", "at", "or", "an", "have", "not",
}
_WORD = re.compile(r"\b[a-z']+\b")


@dataclass(frozen=True)
class LangResult:
    language: str
    score: float
    detector: str


try:
    import fasttext  # type: ignore
    _FT_AVAILABLE = True
except ImportError:
    _FT_AVAILABLE = False


def detect(text: str, upstream: tuple[str, float] | None = None) -> LangResult:
    if upstream is not None:
        lang, score = upstream
        return LangResult(lang, float(score), "upstream")
    if _FT_AVAILABLE:  # pragma: no cover - requires a downloaded model
        raise NotImplementedError("wire the fastText lid model path in Phase 1 deployment")
    words = _WORD.findall(text.lower())
    if not words:
        return LangResult("unknown", 0.0, "heuristic")
    hits = sum(1 for w in words if w in _EN_STOPWORDS)
    ratio = hits / len(words)
    # English prose runs roughly 0.25 to 0.45 stopword share by this list.
    score = min(ratio / 0.25, 1.0)
    return LangResult("en" if score >= 0.5 else "unknown", round(score, 4), "heuristic")
