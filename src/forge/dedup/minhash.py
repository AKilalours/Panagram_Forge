"""Phase 1. MinHash LSH near-duplicate detection. 128 permutations, Jaccard 0.8."""

from __future__ import annotations


def build_index(docs) -> object:  # noqa: ANN001
    raise NotImplementedError("Phase 1")


def find_duplicates(index, doc) -> list[str]:  # noqa: ANN001
    raise NotImplementedError("Phase 1")
