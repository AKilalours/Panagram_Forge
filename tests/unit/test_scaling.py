"""Phase 7 scaling arithmetic.

The first test guards the single most common way a distributed-training benchmark in a
personal project turns out to be wrong.
"""

import pytest

from forge.training.distributed import (
    DistConfig,
    benchmark_matrix,
    deepspeed_config,
    fsdp_plan,
    ray_scaling_config,
)
from forge.training.scaling import (
    ScalingError,
    ThroughputMeasurement,
    assert_global_batch_invariant,
    cost_per_run,
    global_batch,
    grad_accum_for_world_size,
    linear_lr_for_batch,
    mfu,
    scaling_efficiency,
)


def test_batch_size_in_configs_is_per_device_not_global():
    """Move from 1 GPU to 4 and leave the config alone, and the global batch quadruples.
    The run then has different learning dynamics, and any throughput comparison against
    the 1-GPU run is comparing two different experiments."""
    assert global_batch(32, 2, 1) == 64
    assert global_batch(32, 2, 4) == 256


def test_grad_accum_holds_the_global_batch_constant():
    for ws in (1, 2, 4, 8):
        accum = grad_accum_for_world_size(256, 32, ws)
        assert global_batch(32, accum, ws) == 256


def test_a_non_dividing_global_batch_is_refused():
    with pytest.raises(ScalingError, match="not divisible"):
        grad_accum_for_world_size(100, 32, 4)


def test_the_invariance_check_catches_a_broken_benchmark():
    configs = [
        {"world_size": 1, "per_device_batch": 32, "grad_accum": 8},
        {"world_size": 4, "per_device_batch": 32, "grad_accum": 8},   # 4x the global batch
    ]
    with pytest.raises(ScalingError, match="different experiments"):
        assert_global_batch_invariant(configs)


def test_a_correct_benchmark_passes_the_check():
    configs = [
        {"world_size": 1, "per_device_batch": 32, "grad_accum": 8},
        {"world_size": 4, "per_device_batch": 32, "grad_accum": 2},
    ]
    assert assert_global_batch_invariant(configs) == 256


def test_benchmark_matrix_is_batch_invariant_by_construction():
    matrix = benchmark_matrix(256, world_sizes=(1, 2, 4), strategies=("fsdp", "deepspeed"))
    assert len({c.global_batch_size for c in matrix}) == 1
    assert len(matrix) == 6


def test_deepspeed_train_batch_size_is_derived_not_hardcoded():
    """DeepSpeed validates train_batch_size == micro * accum * world internally, so a
    hardcoded value fails at launch or, worse, silently changes the experiment."""
    cfg = DistConfig(world_size=4, per_device_batch=32, grad_accum=2)
    ds = deepspeed_config(cfg, lr=2e-5)
    assert ds["train_batch_size"] == 256
    assert (
        ds["train_batch_size"]
        == ds["train_micro_batch_size_per_gpu"] * ds["gradient_accumulation_steps"] * cfg.world_size
    )


def test_deepspeed_precision_blocks_match_the_config():
    assert deepspeed_config(DistConfig(precision="bf16"), 1e-5)["bf16"]["enabled"]
    assert deepspeed_config(DistConfig(precision="fp16"), 1e-5)["fp16"]["enabled"]
    assert "bf16" not in deepspeed_config(DistConfig(precision="fp32"), 1e-5)


def test_cpu_offload_requires_zero_stage_three():
    with pytest.raises(ValueError):
        deepspeed_config(DistConfig(cpu_offload=True, zero_stage=2), 1e-5)


def test_fsdp_wraps_at_the_layer_boundary():
    """Wrapping the whole model instead of each transformer layer is why FSDP sometimes
    saves no memory at all."""
    assert fsdp_plan(DistConfig())["auto_wrap_policy"] == "transformer_layer"


def test_ray_config_matches_world_size():
    assert ray_scaling_config(DistConfig(world_size=4))["num_workers"] == 4


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        DistConfig(strategy="magic")


# ------------------------------------------------------------------ measurement

def test_throughput_and_scaling_efficiency():
    one = ThroughputMeasurement(1, 100, 200.0, 256, 512)
    four = ThroughputMeasurement(4, 100, 60.0, 256, 512)
    assert one.examples_per_second() == 128.0
    assert abs(scaling_efficiency(one, four) - 0.8333) < 1e-3


def test_scaling_efficiency_requires_a_single_device_baseline():
    with pytest.raises(ScalingError):
        scaling_efficiency(ThroughputMeasurement(2, 10, 1.0, 8, 8),
                           ThroughputMeasurement(4, 10, 1.0, 8, 8))


def test_mfu_above_one_is_rejected_rather_than_reported():
    """A run cannot beat the hardware. An MFU above 1.0 means the FLOPs model or the
    quoted peak is wrong, and reporting it would be reporting a broken measurement."""
    m = ThroughputMeasurement(1, 100, 1.0, 256, 512)
    with pytest.raises(ScalingError, match="exceeds 1.0"):
        mfu(m, flops_per_token=1e12, device_peak_flops=1e9)


def test_mfu_is_a_fraction_for_realistic_inputs():
    """DeBERTa-v3-base sized: hidden 768, 12 layers, 512 tokens, on an A100 bf16 peak."""
    from forge.training.scaling import encoder_flops_per_token

    fpt = encoder_flops_per_token(hidden=768, layers=12, seq_len=512)
    m = ThroughputMeasurement(1, 100, 200.0, 32, 512)
    v = mfu(m, flops_per_token=fpt, device_peak_flops=3.12e14)
    assert 0.0 < v < 1.0, v


def test_flops_model_is_dominated_by_parameters_at_short_sequences():
    from forge.training.scaling import encoder_flops_per_token, encoder_params

    fpt = encoder_flops_per_token(hidden=768, layers=12, seq_len=512)
    param_term = 6 * encoder_params(768, 12)
    assert fpt > param_term
    assert fpt < param_term * 2, "the attention term should not dominate at 512 tokens"


def test_cost_per_run_uses_measured_throughput():
    m = ThroughputMeasurement(4, 100, 60.0, 256, 512)
    c = cost_per_run(m, total_examples=400_000, epochs=3, hourly_rate_per_device=3.5)
    # Fields must reconcile with each other; a report where device_hours does not equal
    # wall_hours * world_size reads as an arithmetic error.
    assert c["device_hours"] == pytest.approx(c["wall_hours"] * c["world_size"], rel=1e-3)
    assert c["usd"] == pytest.approx(c["device_hours"] * 3.5, rel=1e-2)


def test_linear_lr_scaling_is_offered_as_a_heuristic():
    assert linear_lr_for_batch(2e-5, 64, 256) == pytest.approx(8e-5)
