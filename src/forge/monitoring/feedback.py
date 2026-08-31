"""Phase 8. Feedback intake.

Raw thumbs-down never becomes a training label. Users disagree with correct
predictions constantly, and a detector that learns from unverified disagreement learns
to agree with whoever complains loudest. Feedback goes to a verification queue; only
verified labels reach the Failure Atlas.
"""

from __future__ import annotations


def enqueue(sample_id: str, user_verdict: str, text_hash: str) -> None:
    raise NotImplementedError("Phase 8")


def promote_verified(item_id: str, verified_label: str, verifier: str) -> dict:
    raise NotImplementedError("Phase 8")
