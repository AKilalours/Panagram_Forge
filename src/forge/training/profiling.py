"""Phase 7. Profiling.

Rule for this file: only measured numbers get reported. PyTorch Profiler and Nsight
Systems produce traces; the traces go in reports/experiments/ next to the before and
after configuration. "GPU utilization improved" without a trace is not a result.

What gets measured: data loading, tokenization, forward, backward, attention,
collective communication, checkpoint write.
"""

from __future__ import annotations


def profile_step(fn, *args, **kwargs):  # noqa: ANN001, ANN201
    raise NotImplementedError("Phase 7")
