"""Scaling arithmetic for distributed training.

--------------------------------------------------------------------------------
The thing that silently invalidates an FSDP versus DeepSpeed comparison
--------------------------------------------------------------------------------
`batch_size` in every training config is PER DEVICE. So the batch the optimizer
actually sees is

    global_batch = per_device_batch * grad_accum * world_size

Move from 1 GPU to 4 and leave the config alone, and the global batch quadruples. The
run now has a different effective batch size, a different number of optimizer steps per
epoch, and different learning dynamics. Any throughput comparison between two strategies
run that way is comparing two different experiments, and the resulting "DeepSpeed is
faster" or "FSDP converges better" claim is meaningless.

`grad_accum_for_world_size` holds the global batch constant as world size changes, and
`assert_global_batch_invariant` is the check to put in a benchmark harness. This is the
single most common way a distributed-training benchmark in a personal project turns out
to be wrong, and it does not raise an error on its own.

--------------------------------------------------------------------------------
Reporting rules
--------------------------------------------------------------------------------
Only measured numbers go in a report. These functions compute derived quantities FROM
measurements; none of them estimate a measurement. `mfu` needs a real device peak FLOPs
figure and a real step time, and it returns a fraction that should be sanity-checked: an
MFU above 1.0 means the FLOPs model or the peak figure is wrong, not that the run beat
physics, so it raises.
"""

from __future__ import annotations

from dataclasses import dataclass


class ScalingError(ValueError):
    pass


def global_batch(per_device: int, grad_accum: int, world_size: int) -> int:
    if min(per_device, grad_accum, world_size) < 1:
        raise ScalingError("per_device, grad_accum and world_size must all be >= 1")
    return per_device * grad_accum * world_size


def grad_accum_for_world_size(
    target_global_batch: int, per_device: int, world_size: int
) -> int:
    """Accumulation steps that keep the global batch fixed at a new world size."""
    denom = per_device * world_size
    if denom < 1:
        raise ScalingError("per_device and world_size must be >= 1")
    if target_global_batch % denom:
        raise ScalingError(
            f"global batch {target_global_batch} is not divisible by "
            f"per_device({per_device}) * world_size({world_size}) = {denom}. "
            "Pick a global batch that divides cleanly, or the comparison silently runs "
            "at a different effective batch size on each configuration."
        )
    return target_global_batch // denom


def assert_global_batch_invariant(configs: list[dict], label: str = "benchmark") -> int:
    """Every configuration in a benchmark must share one global batch size.

    Put this at the top of any harness comparing strategies or world sizes.
    """
    seen: dict[int, dict] = {}
    for c in configs:
        gb = global_batch(c["per_device_batch"], c["grad_accum"], c["world_size"])
        seen.setdefault(gb, c)
    if len(seen) != 1:
        detail = ", ".join(
            f"{gb} (world_size={c['world_size']}, per_device={c['per_device_batch']}, "
            f"grad_accum={c['grad_accum']})"
            for gb, c in sorted(seen.items())
        )
        raise ScalingError(
            f"{label}: configurations do not share a global batch size: {detail}. "
            "They are different experiments, so any throughput or quality comparison "
            "between them is invalid."
        )
    return next(iter(seen))


def linear_lr_for_batch(base_lr: float, base_global_batch: int, new_global_batch: int) -> float:
    """Linear scaling rule. A heuristic, not a law.

    Reported here so a changed batch size is accompanied by a deliberate learning-rate
    decision rather than an accidental one. It breaks down at very large batch sizes;
    treat it as a starting point to be tuned, not a result.
    """
    if base_global_batch < 1 or new_global_batch < 1:
        raise ScalingError("batch sizes must be >= 1")
    return base_lr * (new_global_batch / base_global_batch)


@dataclass(frozen=True)
class ThroughputMeasurement:
    """All fields MEASURED. Nothing here is estimated."""

    world_size: int
    steps: int
    wall_seconds: float
    global_batch_size: int
    tokens_per_example: int

    def examples_per_second(self) -> float:
        if self.wall_seconds <= 0:
            raise ScalingError("wall_seconds must be positive")
        return self.steps * self.global_batch_size / self.wall_seconds

    def tokens_per_second(self) -> float:
        return self.examples_per_second() * self.tokens_per_example

    def seconds_per_step(self) -> float:
        if self.steps < 1:
            raise ScalingError("steps must be >= 1")
        return self.wall_seconds / self.steps


def scaling_efficiency(single: ThroughputMeasurement, multi: ThroughputMeasurement) -> float:
    """Fraction of linear speedup achieved. 1.0 is perfect, and never happens.

    Below roughly 0.7 says communication or data loading dominates, which is a finding
    worth chasing rather than a number to bury.
    """
    if single.world_size != 1:
        raise ScalingError("the baseline measurement must have world_size == 1")
    speedup = multi.examples_per_second() / single.examples_per_second()
    return speedup / multi.world_size


def encoder_params(hidden: int, layers: int, vocab: int = 0) -> int:
    """Approximate non-embedding parameter count of a transformer encoder.

    12 * layers * hidden^2 covers attention projections (4 * h^2) and the MLP
    (8 * h^2 at the usual 4x expansion). Embeddings are added separately because they
    contribute parameters but almost no FLOPs per token.
    """
    return 12 * layers * hidden**2 + vocab * hidden


def encoder_flops_per_token(hidden: int, layers: int, seq_len: int, vocab: int = 0) -> float:
    """Approximate forward+backward FLOPs per token.

    Two terms:
      6 * N          the standard approximation for N non-embedding parameters
                     (2 forward, 4 backward, per parameter per token)
      12 * L * h * s the attention score/context term, which the parameter count misses
                     entirely and which grows with sequence length

    Approximate on purpose and labelled as such: this feeds MFU, which is a diagnostic
    for finding bottlenecks, not a result to report on its own.
    """
    n = encoder_params(hidden, layers, vocab=0)
    attention = 12 * layers * hidden * seq_len
    return 6 * n + attention


def mfu(
    measurement: ThroughputMeasurement,
    flops_per_token: float,
    device_peak_flops: float,
) -> float:
    """Model FLOPs Utilisation. device_peak_flops must be the real number for the real
    dtype: quoting a bf16 tensor-core peak for an fp32 run inflates the denominator and
    makes utilisation look worse than it is."""
    if device_peak_flops <= 0:
        raise ScalingError("device_peak_flops must be positive")
    achieved = measurement.tokens_per_second() * flops_per_token
    total_peak = device_peak_flops * measurement.world_size
    frac = achieved / total_peak
    if frac > 1.0:
        raise ScalingError(
            f"computed MFU of {frac:.2f} exceeds 1.0. The FLOPs model or the quoted peak "
            "is wrong; a run cannot exceed the hardware. Fix the inputs rather than "
            "reporting the number."
        )
    return frac


def cost_per_run(
    measurement: ThroughputMeasurement,
    total_examples: int,
    epochs: int,
    hourly_rate_per_device: float,
) -> dict:
    """Cost of a training run from measured throughput.

    Worth far more in a writeup than a list of service names: it is the number that shows
    the infrastructure was actually operated rather than described.
    """
    if hourly_rate_per_device < 0:
        raise ScalingError("hourly rate cannot be negative")
    seconds = total_examples * epochs / measurement.examples_per_second()
    hours = seconds / 3600
    device_hours = hours * measurement.world_size
    # Rounded from the same unrounded quantities so the fields reconcile with each other.
    # Rounding wall_hours first and multiplying gives a device_hours that does not match
    # wall_hours * world_size, which reads as an arithmetic error in a report.
    return {
        "wall_hours": round(hours, 4),
        "device_hours": round(device_hours, 4),
        "usd": round(device_hours * hourly_rate_per_device, 2),
        "examples_per_second": round(measurement.examples_per_second(), 2),
        "world_size": measurement.world_size,
    }
