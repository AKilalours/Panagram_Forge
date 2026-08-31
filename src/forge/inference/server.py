"""Phase 8. Detector serving.

Explicitly not vLLM. vLLM's value is continuous batching and paged KV cache for
autoregressive decoding; FORGE-Base is a bidirectional encoder with no KV cache and a
fixed 512-token window, so none of that applies. vLLM is used on the generation side
instead (src/forge/generation/generators/base.py).

Serving path: FastAPI, dynamic batching, ONNX Runtime if it measurably wins,
TensorRT only if the latency budget forces it and the added build complexity is
justified by numbers.
"""

from __future__ import annotations


class Detector:
    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        raise NotImplementedError("Phase 8")

    def predict(self, text: str) -> dict:
        raise NotImplementedError("Phase 8")
