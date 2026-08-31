"""Phase 3 and 7. Training entrypoint.

Single GPU first. Distribution is added in Phase 7 with FSDP as the default and
DeepSpeed ZeRO-3 benchmarked against it, so the choice is measured rather than
asserted. Ray orchestrates multi-node runs and the mining sweeps.
"""

from __future__ import annotations


def run(config: dict) -> None:
    raise NotImplementedError("Phase 3")
