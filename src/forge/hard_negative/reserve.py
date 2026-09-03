"""Read the reserve pool off disk, and score it with a trained checkpoint.

Two small pieces that the CLI needs and that nothing else owned, kept out of `run.py` so
the mining round stays testable against a fake scorer with known failure structure.

**The reserve pool is disjoint from training by construction** (see
`forge.ingestion.run.build`), which is the whole reason mining scans it: a false positive on
a document the model was trained on tells you nothing about production behaviour. This module
re-checks that invariant rather than trusting it, because the cost of getting it wrong is a
"hard negative" set made of memorized training text, which would look like it worked.
"""

from __future__ import annotations

import glob
from pathlib import Path

from forge.hard_negative.mining import ReserveDoc

COLUMNS = ["doc_id", "source_group_id", "text", "domain", "source", "register"]


def load_reserve(root: str | Path, limit: int | None = None) -> list[ReserveDoc]:
    """Load reserve documents from a parquet root. Ordered by doc_id, so runs repeat."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(
            f"no parquet under {root}. The reserve pool is written by ingestion with "
            "with_reserve=True; without it there is nothing to mine."
        )

    docs: list[ReserveDoc] = []
    for f in files:
        available = set(pq.ParquetFile(f).schema.names)
        missing = {"doc_id", "source_group_id", "text"} - available
        if missing:
            raise ValueError(f"{f} is missing required columns {sorted(missing)}")
        for row in pq.read_table(f, columns=[c for c in COLUMNS if c in available]).to_pylist():
            if not (row.get("text") or "").strip():
                continue
            docs.append(ReserveDoc(
                doc_id=row["doc_id"],
                source_group_id=row["source_group_id"],
                text=row["text"],
                source=row.get("source") or "unknown",
                domain=row.get("domain") or "unknown",
                register=row.get("register") or "informational",
            ))

    docs.sort(key=lambda d: d.doc_id)
    return docs[:limit] if limit else docs


def assert_disjoint_from_training(reserve: list[ReserveDoc], training_group_ids) -> None:
    """Refuse to mine a reserve pool that overlaps training.

    Checked on `source_group_id`, not `doc_id`, because a human document and everything
    generated from it share a group and land in the same split. An overlap on the group is
    an overlap on the split, even when no document id repeats.
    """
    training = set(training_group_ids)
    if not training:
        return
    overlap = {d.source_group_id for d in reserve} & training
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} reserve source groups also appear in training data. Mining "
            "these would collect memorization, not production failures, and the resulting "
            f"hard negatives would look effective for the wrong reason. Example: "
            f"{sorted(overlap)[:3]}"
        )


class CheckpointScorer:
    """A `mining.Scorer` backed by a trained arm. Batched, CPU or GPU.

    Scores whole documents the way evaluation does: windows, mean over windows. Mining looks
    for documents the detector calls AI at high confidence, so the aggregation has to match
    the one whose false positives are being characterised. Using max here would mine
    documents that merely contain one odd passage.
    """

    def __init__(self, arm_name: str = "mirror", batch_size: int = 32) -> None:
        from forge.inference.scorer import load_arm

        self._arm = load_arm(arm_name)
        self.batch_size = batch_size
        self.model_version = self._arm.policy.model_version
        self.threshold = self._arm.policy.threshold

    def score(self, texts: list[str]) -> list[float]:
        return [self._arm.score(t).mean if t and t.strip() else 0.0 for t in texts]
