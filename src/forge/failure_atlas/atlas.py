"""Phase 4. The Failure Atlas.

Every failure, from evaluation or from production, becomes a FailureRecord. Records
are embedded and clustered so that failures are described as modes rather than as a
list. "3,000 false positives" tells you nothing. "Cluster 17: citation-heavy academic
prose with long paragraphs, 32 percent of all false positives" tells you what data to
generate next.

This is also what closes the loop: clusters are the input to targeted generation.
"""

from __future__ import annotations


def embed(records: list[dict], model: str) -> list[list[float]]:
    raise NotImplementedError("Phase 4")


def cluster(embeddings, method: str = "hdbscan", min_cluster_size: int = 25):  # noqa: ANN001, ANN201
    raise NotImplementedError("Phase 4")


def summarize(records: list[dict], labels) -> dict:  # noqa: ANN001
    raise NotImplementedError("Phase 4")
