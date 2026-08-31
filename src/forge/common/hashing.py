from __future__ import annotations

import hashlib


def content_sha256(text: str) -> str:
    """Hash of normalized content. Used for exact dedup and for provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
