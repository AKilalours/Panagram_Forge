"""Language identification. Stage 2 of the cleaning pipeline.

Three sources of truth, tried in order, and the function NEVER raises:

  1. **Upstream score.** FineWeb and FineWeb-Edu already carry `language` and
     `language_score` from their own fastText pass. Recomputing that would be work to
     reproduce a field the dataset hands us.
  2. **A real fastText model**, if one is actually loadable. This requires both the
     library AND a model file on disk, pointed at by FORGE_FASTTEXT_LID_PATH.
  3. **A stopword heuristic.** Crude, and it records `detector: "heuristic"` on the
     record so nothing downstream mistakes it for a real score.

--------------------------------------------------------------------------------
Why this file was rewritten
--------------------------------------------------------------------------------
It previously raised NotImplementedError whenever the fastText LIBRARY happened to be
importable, on the reasoning that a model path still had to be wired. That is a
"library present" check standing in for a "model usable" check, and the two are not the
same thing.

The consequence was environment-dependent behaviour from identical code. On a laptop
where the import failed, the heuristic ran and everything passed. On the GPU pod, where
the `data` extra pulls in a package providing `fasttext`, the same call raised. Three
tests failed there and passed locally, and more importantly `forge ingest` would have
crashed on the first document from any source without an upstream score, which is every
source except FineWeb.

Capability checks must test the capability, not a proxy for it. This module now asks
whether a model actually loads, and degrades to the heuristic when it does not.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as",
    "was", "on", "be", "by", "this", "are", "from", "at", "or", "an", "have", "not",
}
_WORD = re.compile(r"\b[a-z']+\b")

# Env var pointing at a fastText lid model (lid.176.bin or similar).
MODEL_PATH_ENV = "FORGE_FASTTEXT_LID_PATH"

_model = None
_load_attempted = False


@dataclass(frozen=True)
class LangResult:
    language: str
    score: float
    detector: str


def _fasttext_model():
    """Load the model once, or return None. Never raises: a missing or broken model is
    a reason to fall back, not a reason to stop ingesting a corpus."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    path = os.getenv(MODEL_PATH_ENV)
    if not path or not os.path.exists(path):
        return None
    try:
        import fasttext

        _model = fasttext.load_model(path)
    except Exception:
        _model = None
    return _model


def reset_model_cache() -> None:
    """For tests. Lets a test exercise both branches in one process."""
    global _model, _load_attempted
    _model, _load_attempted = None, False


def _heuristic(text: str) -> LangResult:
    words = _WORD.findall(text.lower())
    if not words:
        return LangResult("unknown", 0.0, "heuristic")
    ratio = sum(1 for w in words if w in _EN_STOPWORDS) / len(words)
    # English prose runs roughly 0.25 to 0.45 stopword share by this list.
    score = min(ratio / 0.25, 1.0)
    return LangResult("en" if score >= 0.5 else "unknown", round(score, 4), "heuristic")


def detect(text: str, upstream: tuple[str, float] | None = None) -> LangResult:
    if upstream is not None:
        lang, score = upstream
        return LangResult(lang, float(score), "upstream")

    model = _fasttext_model()
    if model is not None:
        try:
            labels, probs = model.predict(text.replace("\n", " ")[:2000], k=1)
            return LangResult(labels[0].replace("__label__", ""), float(probs[0]), "fasttext")
        except Exception:
            # A model that loaded but fails on a specific document is not a reason to
            # abandon the run.
            pass

    return _heuristic(text)
