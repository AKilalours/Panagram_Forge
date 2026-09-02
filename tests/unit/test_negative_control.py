"""The negative control must catch a leaking pipeline and must not cry wolf.

A human/AI image detector can score near-perfectly by separating two encoding pipelines
rather than photographs from generations. The control labels HUMAN images arbitrarily and
checks that a model cannot beat chance on them. If it can, the model is reading something
about how the images were produced, and every number from the real experiment is void.

These tests exercise the control's logic with an injected trainer, so they run without a
GPU. What is under test is that the labels are arbitrary but reproducible, that the
threshold accounts for sampling noise, and that a leak is actually caught.
"""

from __future__ import annotations

from collections import Counter

import pytest

from forge.image.negative_control import (
    ControlResult,
    NegativeControlFailed,
    arbitrary_label,
    arbitrary_labels,
    assert_negative_control,
    chance_margin,
    run_negative_control,
)

IDS = [f"img_{i:05d}" for i in range(4000)]


def test_labels_are_balanced() -> None:
    counts = Counter(arbitrary_labels(IDS).values())
    assert abs(counts[0] - counts[1]) < 0.05 * len(IDS), counts


def test_labels_are_reproducible() -> None:
    """A failed control must be reproducible exactly, or it cannot be debugged."""
    assert arbitrary_labels(IDS) == arbitrary_labels(IDS)


def test_labels_change_with_the_salt() -> None:
    """Different salts give an independent draw, so the control can be repeated."""
    a = arbitrary_labels(IDS)
    b = arbitrary_labels(IDS, salt="other")
    disagreements = sum(1 for i in IDS if a[i] != b[i])
    assert 0.4 * len(IDS) < disagreements < 0.6 * len(IDS)


def test_labels_carry_no_information_about_the_id_prefix() -> None:
    """If ids were labelled by their order, the control would test ordering, not leakage."""
    first_half = [arbitrary_label(i) for i in IDS[: len(IDS) // 2]]
    second_half = [arbitrary_label(i) for i in IDS[len(IDS) // 2 :]]
    assert abs(sum(first_half) - sum(second_half)) < 0.1 * len(IDS)


def test_a_clean_pipeline_passes() -> None:
    result = run_negative_control(IDS, train_and_score=lambda labels: 0.502)
    assert result.passed
    assert "PASSED" in result.explain()
    assert_negative_control(result)


def test_a_leaking_pipeline_fails() -> None:
    """The case this exists for: the model separates human images from human images."""
    result = run_negative_control(IDS, train_and_score=lambda labels: 0.83)
    assert not result.passed
    with pytest.raises(NegativeControlFailed) as e:
        assert_negative_control(result)
    assert "void" in str(e.value)
    assert "normalis" in str(e.value), "the message should point at the likeliest cause"


def test_the_margin_shrinks_as_the_sample_grows() -> None:
    """Chance drift on 100 images is not chance drift on 100,000."""
    assert chance_margin(100) > chance_margin(10_000) > chance_margin(1_000_000)


def test_a_small_sample_is_not_flagged_for_ordinary_noise() -> None:
    """A control that fails spuriously wastes an afternoon; keep the bias conservative."""
    result = run_negative_control(IDS[:200], train_and_score=lambda labels: 0.60)
    assert result.passed, "0.60 on 200 samples is within ordinary noise"


def test_a_large_sample_flags_a_small_but_real_effect() -> None:
    """On enough data, even a modest edge means a real signature."""
    result = run_negative_control(IDS, train_and_score=lambda labels: 0.56)
    assert not result.passed


def test_the_trainer_receives_the_arbitrary_labels() -> None:
    seen: dict = {}

    def trainer(labels: dict) -> float:
        seen.update(labels)
        return 0.5

    run_negative_control(IDS[:100], train_and_score=trainer)
    assert seen == arbitrary_labels(IDS[:100])


def test_too_few_images_is_an_error_not_a_pass() -> None:
    """Silently passing on a degenerate sample would be the worst possible behaviour."""
    with pytest.raises(ValueError):
        run_negative_control(["only_one"], train_and_score=lambda labels: 0.5)


def test_result_reports_what_it_measured() -> None:
    result = ControlResult(auroc=0.51, n=1000, margin=chance_margin(1000))
    assert "1000" in result.explain()
    assert "0.51" in result.explain()
