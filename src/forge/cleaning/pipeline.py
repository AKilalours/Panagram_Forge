"""Phase 1. The fixed-order cleaning pipeline from data_spec_v1 section 7.

Order matters in one place especially: deduplication runs BEFORE split assignment.
Deduplicating afterwards leaves near-duplicate pairs straddling train and test, which
is the same leak as a row-level split wearing a different hat.
"""

from __future__ import annotations

STAGES = [
    "schema_validation",
    "language_id",
    "unicode_normalization",
    "markup_removal",
    "length_filter",
    "quality_scoring",
    "pii_filter",
    "exact_dedup",
    "near_duplicate_dedup",
    "domain_classification",
    "split_assignment",
    "parquet_write",
]


def run(config: dict) -> None:
    raise NotImplementedError("Phase 1")
