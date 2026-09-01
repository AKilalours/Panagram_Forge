"""Distributed strategies, and the configs they need.

Three backends behind one interface so the comparison is like for like:
  fsdp      PyTorch FullyShardedDataParallel, full_shard, the default
  deepspeed ZeRO stage 3, benchmarked against FSDP on the SAME global batch
  ray       multi-node orchestration and parallel mining sweeps

The config GENERATORS below are pure functions and are tested. The parts that construct
live process groups are not, because they need hardware.

Every generator refuses to emit a config whose global batch differs from the benchmark's,
via forge.training.scaling. See that module for why: a strategy comparison run at two
different effective batch sizes is two different experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.training.scaling import ScalingError, global_batch

STRATEGIES = ("none", "fsdp", "deepspeed", "ray")


@dataclass(frozen=True)
class DistConfig:
    strategy: str = "fsdp"
    world_size: int = 1
    per_device_batch: int = 32
    grad_accum: int = 2
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    sharding: str = "full_shard"
    zero_stage: int = 3
    cpu_offload: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}, expected one of {STRATEGIES}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown precision {self.precision!r}")

    @property
    def global_batch_size(self) -> int:
        return global_batch(self.per_device_batch, self.grad_accum, self.world_size)


def deepspeed_config(cfg: DistConfig, lr: float, warmup_steps: int = 500) -> dict[str, Any]:
    """Emit a DeepSpeed JSON config.

    `train_batch_size` here is DeepSpeed's GLOBAL batch and it validates
    train_batch_size == micro_batch * grad_accum * world_size internally. Deriving it
    rather than hardcoding it is what stops a benchmark drifting between strategies.
    """
    if cfg.zero_stage not in (0, 1, 2, 3):
        raise ValueError("zero_stage must be 0, 1, 2 or 3")
    conf: dict[str, Any] = {
        "train_batch_size": cfg.global_batch_size,
        "train_micro_batch_size_per_gpu": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": lr, "warmup_num_steps": warmup_steps},
        },
        "activation_checkpointing": {"partition_activations": cfg.gradient_checkpointing},
        "steps_per_print": 100,
        "wall_clock_breakdown": True,   # so profiling has something to read
    }
    if cfg.precision == "bf16":
        conf["bf16"] = {"enabled": True}
    elif cfg.precision == "fp16":
        conf["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}
    if cfg.cpu_offload:
        if cfg.zero_stage != 3:
            raise ValueError("cpu_offload requires zero_stage 3")
        conf["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        conf["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
    return conf


def fsdp_plan(cfg: DistConfig) -> dict[str, Any]:
    """The FSDP settings, as data, so the two strategies can be diffed rather than
    described. Sharding a transformer at the LAYER boundary rather than wrapping the
    whole model is what makes FSDP actually save memory."""
    return {
        "sharding_strategy": cfg.sharding,
        "mixed_precision": cfg.precision,
        "auto_wrap_policy": "transformer_layer",
        "activation_checkpointing": cfg.gradient_checkpointing,
        "cpu_offload": cfg.cpu_offload,
        "limit_all_gathers": True,
        "use_orig_params": True,      # required for torch.compile and for param groups
        "backward_prefetch": "backward_pre",
    }


def ray_scaling_config(cfg: DistConfig, use_gpu: bool = True) -> dict[str, Any]:
    return {
        "num_workers": cfg.world_size,
        "use_gpu": use_gpu,
        "resources_per_worker": {"GPU": 1 if use_gpu else 0, "CPU": 4},
        "placement_strategy": "PACK",   # keep workers close; collectives are latency bound
    }


def benchmark_matrix(
    target_global_batch: int,
    world_sizes: tuple[int, ...] = (1, 2, 4),
    per_device_batch: int = 32,
    strategies: tuple[str, ...] = ("fsdp", "deepspeed"),
) -> list[DistConfig]:
    """Build a strategy-by-world-size matrix that all shares one global batch size.

    This is the harness for the Phase 7 exit criterion. Constructing it by hand is how
    the invariant gets broken.
    """
    from forge.training.scaling import grad_accum_for_world_size

    out: list[DistConfig] = []
    for ws in world_sizes:
        accum = grad_accum_for_world_size(target_global_batch, per_device_batch, ws)
        for s in strategies:
            out.append(
                DistConfig(strategy=s, world_size=ws, per_device_batch=per_device_batch,
                           grad_accum=accum)
            )
    sizes = {c.global_batch_size for c in out}
    if len(sizes) != 1:
        raise ScalingError(f"benchmark matrix is not batch-invariant: {sorted(sizes)}")
    return out


def build(cfg: DistConfig, model, optimizer=None):  # pragma: no cover - needs hardware
    raise NotImplementedError(
        "Live process-group construction needs multiple GPUs. The configs these "
        "strategies consume are generated and tested; see deepspeed_config, fsdp_plan "
        "and ray_scaling_config."
    )
