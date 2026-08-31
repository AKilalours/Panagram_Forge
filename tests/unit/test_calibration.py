import numpy as np

from forge.evaluation.calibration import ai_probability, apply_temperature, fit_temperature
from forge.evaluation.metrics import expected_calibration_error


def _overconfident(n=4000, scale=3.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    base = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 0.8, n)
    return np.stack([-base * scale, base * scale], 1), labels


def test_temperature_scaling_reduces_ece():
    logits, labels = _overconfident()
    before = expected_calibration_error(labels, apply_temperature(logits, 1.0)[:, 1])
    t = fit_temperature(logits, labels)
    after = expected_calibration_error(labels, apply_temperature(logits, t)[:, 1])
    assert after < before / 2, f"ECE {before} -> {after}"


def test_overconfident_model_gets_a_temperature_above_one():
    logits, labels = _overconfident()
    assert fit_temperature(logits, labels) > 1.0


def test_temperature_preserves_ranking_so_auroc_is_unchanged():
    """Calibration must not change WHICH documents look most AI, only how confident the
    number is. If it changed ranking it would be doing something other than calibrating."""
    from forge.evaluation.metrics import auroc

    logits, labels = _overconfident()
    a = auroc(labels, ai_probability(logits, 1.0))
    b = auroc(labels, ai_probability(logits, fit_temperature(logits, labels)))
    assert abs(a - b) < 1e-9


def test_zero_temperature_is_rejected():
    try:
        apply_temperature(np.zeros((2, 2)), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
