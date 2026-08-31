import math

from forge.evaluation import metrics as m


def test_fpr_counts_accused_humans_only():
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.3, 0.8, 0.9]
    assert math.isclose(m.false_positive_rate(y_true, y_score), 0.25)


def test_fnr():
    y_true = [0, 0, 1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9, 0.1, 0.9]
    assert math.isclose(m.false_negative_rate(y_true, y_score), 0.25)


def test_threshold_at_fpr_respects_the_budget():
    y_true = [0] * 1000 + [1] * 100
    y_score = [i / 1000 for i in range(1000)] + [0.99] * 100
    thr = m.threshold_at_fpr(y_true, y_score, target_fpr=0.01)
    assert m.false_positive_rate(y_true, y_score, thr) <= 0.01


def test_auroc_perfect_and_random():
    assert math.isclose(m.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
    assert math.isclose(m.auroc([0, 1], [0.5, 0.5]), 0.5)


def test_ece_is_zero_for_a_perfectly_confident_correct_model():
    y_true = [0, 0, 1, 1]
    y_score = [0.0, 0.0, 1.0, 1.0]
    assert m.expected_calibration_error(y_true, y_score) < 1e-9


def test_data_efficiency_rejects_zero_examples():
    try:
        m.data_efficiency(0.05, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
