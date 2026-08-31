"""Phase 7. Distributed strategies.

Three backends, one interface, so the comparison is apples to apples:
  - fsdp     : PyTorch FullyShardedDataParallel, full_shard, the default
  - deepspeed: ZeRO stage 3, benchmarked against FSDP on the same config
  - ray      : multi-node orchestration and parallel mining sweeps

Also here: mixed precision (bf16), gradient accumulation, activation checkpointing,
distributed samplers, and checkpoint resume. Resume matters more than it looks: a
multi-hour run that cannot resume turns every preemption into a lost day.
"""

from __future__ import annotations

STRATEGIES = ("none", "fsdp", "deepspeed", "ray")


def build(strategy: str, config: dict):  # noqa: ANN201
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    raise NotImplementedError("Phase 7")
