"""Group-aware, deterministic split assignment.

Why this file exists at all: the single easiest way to produce a fake-good result in
this project is to random-split rows after generating mirrors. A human document and
its mirrors are near-identical in topic and structure, so a row split puts nearly the
same content in train and test and every metric jumps. Splitting on the group id
prevents that, and hashing (rather than shuffling) means re-running ingestion on more
data keeps every old document in its original split, so dataset v0.2 stays comparable
to v0.1.
"""

from __future__ import annotations

import hashlib

from forge.common.schemas import Split

SPLIT_SALT = "forge-v1"
RATIOS: dict[Split, float] = {Split.TRAIN: 0.80, Split.VAL: 0.10, Split.TEST: 0.10}

_BUCKETS = 10_000


def group_id_for(doc_id: str) -> str:
    """Every derived record (mirror, hard negative, adversarial variant) inherits this."""
    return f"grp_{doc_id}"


def assign_split(source_group_id: str, salt: str = SPLIT_SALT) -> Split:
    digest = hashlib.sha256(f"{source_group_id}{salt}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % _BUCKETS
    train_cut = int(RATIOS[Split.TRAIN] * _BUCKETS)
    val_cut = train_cut + int(RATIOS[Split.VAL] * _BUCKETS)
    if bucket < train_cut:
        return Split.TRAIN
    if bucket < val_cut:
        return Split.VAL
    return Split.TEST


def check_no_group_leakage(records: list[tuple[str, Split]]) -> None:
    """Raise if any source_group_id appears in more than one split."""
    seen: dict[str, Split] = {}
    for group, split in records:
        prior = seen.setdefault(group, split)
        if prior is not split:
            raise ValueError(
                f"group leakage: {group} appears in both {prior.value} and {split.value}"
            )
