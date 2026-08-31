"""Phase 4. Hard negative mining. This is the heart of FORGE.

    human reserve pool -> current detector -> high-confidence false positives
      -> rank -> embed -> cluster -> sample across clusters -> targeted mirrors
      -> retrain

One detail that decides whether this works: sample proportionally across clusters
rather than taking the global top-k by confidence. Top-k spends the whole budget on
whichever single failure mode happens to produce the most confident errors, and the
model overfits to it while the other failure modes go untouched.

The reserve pool is never trained on directly. Only documents selected by mining, and
the mirrors generated from them, enter the training set.
"""

from __future__ import annotations


def scan(model_version: str, reserve_path: str, limit: int) -> list[dict]:
    raise NotImplementedError("Phase 4")


def select(failures: list[dict], min_confidence: float, max_selected: int) -> list[dict]:
    raise NotImplementedError("Phase 4")
