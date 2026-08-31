"""Exact deduplication on the content hash. Stage 8."""

from __future__ import annotations

from forge.common.hashing import content_sha256


class ExactDeduper:
    def __init__(self) -> None:
        self._seen: dict[str, str] = {}  # hash -> first doc_id that had it

    def is_duplicate(self, doc_id: str, text: str) -> str | None:
        """Return the doc_id of the original if this is a repeat, else None."""
        h = content_sha256(text)
        first = self._seen.get(h)
        if first is not None:
            return first
        self._seen[h] = doc_id
        return None

    def __len__(self) -> int:
        return len(self._seen)
