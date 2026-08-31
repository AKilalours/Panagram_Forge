"""Phase 7. Distributed scan of the human reserve pool.

Why Spark here and nowhere else: this stage runs the current detector over roughly 5
million documents and keeps only the high-confidence false positives. It is
embarrassingly parallel over shards, it is the largest data job in the project, and it
runs repeatedly (once per flywheel turn). Phase 1's 400k-document cleaning job runs on
Polars on one machine because Spark there would be operational cost for no throughput.
"""

from __future__ import annotations


def run(reserve_path: str, model_uri: str, min_confidence: float = 0.90) -> None:
    raise NotImplementedError("Phase 7")
