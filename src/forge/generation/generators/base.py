"""Phase 2. Generator backends. vLLM for open-weight batch generation, HTTP for API models.

Note the asymmetry with serving: vLLM belongs here, on the generation side, where the
workload is autoregressive decoding at volume. It does not belong on the detector's
serving path, which runs a bidirectional encoder.
"""

from __future__ import annotations

from typing import Protocol


class Generator(Protocol):
    family: str
    model_id: str
    revision: str

    def generate(self, prompts: list[str], **decoding) -> list[str]: ...


class VLLMGenerator:
    """Offline batch generation for open-weight families."""

    def __init__(self, model_id: str, revision: str, tensor_parallel_size: int = 1) -> None:
        raise NotImplementedError("Phase 2")


class APIGenerator:
    """Held-out frontier model. Records the served model string per request."""

    def __init__(self, endpoint: str) -> None:
        raise NotImplementedError("Phase 2")
