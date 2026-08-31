"""Phase 8. Dynamic batching: collect requests for a few milliseconds, run one batch.

The tradeoff is explicit. Waiting adds latency to every request and multiplies
throughput. The wait window is tuned against the P95 latency budget in the release
gate, not chosen by feel.
"""

from __future__ import annotations


class DynamicBatcher:
    def __init__(self, max_batch: int = 32, max_wait_ms: int = 10) -> None:
        raise NotImplementedError("Phase 8")
