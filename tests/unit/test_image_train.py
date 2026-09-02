"""Assembling the image training set must not quietly break the comparison.

WHY THESE TESTS EXIST. Every earlier stage of the image track has its own tests, and all of
them can pass while the assembled dataset is wrong: the arms can drift apart, a group can
straddle a split, a composite can lose its mask, a budget cut can land off by one. None of
those raise. They change the numbers and say nothing.

The test this file is built around is `test_equal_counts_do_not_make_arms_comparable`. Count
equality is the check that feels sufficient and is not, and it is the same shape as every
bug already found in this project: something that reports success while measuring the wrong
thing.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from forge.image.train import (
    ARM_MIRROR,
    ARM_UNMATCHED,
    KIND_COMPOSITE_AI,
    KIND_COMPOSITE_HUMAN,
    KIND_HUMAN,
    KIND_MIRROR,
    KIND_UNMATCHED,
    SHARED,
    ImageSample,
    IndexAssemblyError,
    ai_budget,
    assert_arms_comparable,
    assert_both_classes,
    assert_no_group_leakage,
    build_arm,
    cap_by_budget,
    index_summary,
    patch_iou,
    read_index,
    write_index,
)

SPLITS = ("train", "val", "test")


def _human(i: int, split: str | None = None) -> ImageSample:
    split = split or SPLITS[i % 3]
    return ImageSample(f"img{i:05d}", f"grp{i:05d}", split, 0, KIND_HUMAN, f"px/img{i:05d}")


def _generated(i: int, kind: str, split: str | None = None) -> ImageSample:
    split = split or SPLITS[i % 3]
    return ImageSample(
        f"{kind}_{i:05d}", f"grp{i:05d}", split, 1, kind, f"px/{kind}_{i:05d}"
    )


def _composite(i: int, kind: str) -> ImageSample:
    return ImageSample(
        f"comp_{kind}_{i:05d}",
        f"grp{i:05d}",
        SPLITS[i % 3],
        1 if kind == KIND_COMPOSITE_AI else 0,
        kind,
        f"px/comp_{i:05d}",
        mask_ref=f"mask/comp_{i:05d}",
    )


def _shared(n: int = 300) -> list[ImageSample]:
    out = [_human(i) for i in range(n)]
    out += [_composite(i, KIND_COMPOSITE_AI) for i in range(n // 4)]
    out += [_composite(i, KIND_COMPOSITE_HUMAN) for i in range(n // 4)]
    return out


# --------------------------------------------------------------------------- sample shape


def test_a_composite_without_a_mask_is_refused() -> None:
    """The local head's only ground truth is composite masks. A missing one is a silent hole."""
    with pytest.raises(ValueError, match="mask_ref"):
        ImageSample("c1", "g1", "train", 1, KIND_COMPOSITE_AI, "px/c1")


def test_whole_image_samples_need_no_mask() -> None:
    assert ImageSample("i1", "g1", "train", 0, KIND_HUMAN, "px/i1").mask_ref is None


@pytest.mark.parametrize("kind", [KIND_HUMAN, KIND_MIRROR, KIND_UNMATCHED])
def test_unknown_kinds_are_refused(kind: str) -> None:
    ImageSample("i1", "g1", "train", 0, kind, "px/i1")  # these are known
    with pytest.raises(ValueError, match="unknown sample kind"):
        ImageSample("i1", "g1", "train", 0, "photorealistic", "px/i1")


def test_labels_outside_zero_and_one_are_refused() -> None:
    with pytest.raises(ValueError, match="label"):
        ImageSample("i1", "g1", "train", 2, KIND_HUMAN, "px/i1")


def test_arm_membership_follows_from_kind() -> None:
    assert _human(0).arm == SHARED
    assert _composite(0, KIND_COMPOSITE_AI).arm == SHARED
    assert _generated(0, KIND_MIRROR).arm == ARM_MIRROR
    assert _generated(0, KIND_UNMATCHED).arm == ARM_UNMATCHED


# ------------------------------------------------------------------------------- budgeting


@pytest.mark.parametrize("budget", [1, 7, 50, 199])
def test_the_cut_returns_exactly_the_budget(budget: int) -> None:
    """Off-by-one here becomes an unequal-arms failure two functions away."""
    pool = [_generated(i, KIND_MIRROR) for i in range(200)]
    assert len(cap_by_budget(pool, budget)) == budget


def test_a_budget_larger_than_the_pool_keeps_everything() -> None:
    pool = [_generated(i, KIND_MIRROR) for i in range(20)]
    assert len(cap_by_budget(pool, 999)) == 20


def test_the_cut_draws_from_every_split() -> None:
    pool = [_generated(i, KIND_MIRROR) for i in range(300)]
    got = {s.split for s in cap_by_budget(pool, 60)}
    assert got == set(SPLITS), f"the cut collapsed onto {got}"


def test_the_cut_is_deterministic() -> None:
    pool = [_generated(i, KIND_MIRROR) for i in range(300)]
    first = [s.sample_id for s in cap_by_budget(pool, 90)]
    assert first == [s.sample_id for s in cap_by_budget(pool, 90)]


def test_the_cut_does_not_depend_on_input_order() -> None:
    """Two runs that wrote their parquet files in a different order must build one dataset."""
    pool = [_generated(i, KIND_MIRROR) for i in range(300)]
    shuffled = pool[150:] + pool[:150]
    assert sorted(s.sample_id for s in cap_by_budget(pool, 90)) == sorted(
        s.sample_id for s in cap_by_budget(shuffled, 90)
    )


def test_smaller_cuts_nest_inside_larger_ones() -> None:
    """So a small probe is a genuine preview of the run that follows it."""
    pool = [_generated(i, KIND_MIRROR) for i in range(300)]
    small = {s.sample_id for s in cap_by_budget(pool, 60)}
    large = {s.sample_id for s in cap_by_budget(pool, 180)}
    assert small <= large


def test_the_cut_keeps_input_order() -> None:
    pool = [_generated(i, KIND_MIRROR) for i in range(300)]
    order = {s.sample_id: i for i, s in enumerate(pool)}
    picked = [order[s.sample_id] for s in cap_by_budget(pool, 90)]
    assert picked == sorted(picked)


def test_composites_do_not_count_against_the_generated_budget() -> None:
    """They are supervision, not treatment. Counting them would make the budgets wrong."""
    samples = _shared(100) + [_generated(i, KIND_MIRROR) for i in range(40)]
    assert ai_budget(samples) == 40


# ------------------------------------------------------------------------------ arm shapes


def test_a_treatment_sample_in_the_shared_pool_is_refused() -> None:
    with pytest.raises(IndexAssemblyError, match="shared pool"):
        build_arm(_shared(20) + [_generated(0, KIND_MIRROR)], [_generated(1, KIND_MIRROR)])


def test_an_arm_cannot_mix_two_treatments() -> None:
    treatment = [_generated(0, KIND_MIRROR), _generated(1, KIND_UNMATCHED)]
    with pytest.raises(IndexAssemblyError, match="mixes arms"):
        build_arm(_shared(20), treatment)


def test_properly_built_arms_are_comparable() -> None:
    shared = _shared(200)
    mirrors = [_generated(i, KIND_MIRROR) for i in range(120)]
    randoms = [_generated(i, KIND_UNMATCHED) for i in range(180)]
    budget = min(len(mirrors), len(randoms))
    assert_arms_comparable(
        build_arm(shared, mirrors, budget), build_arm(shared, randoms, budget)
    )


def test_equal_counts_do_not_make_arms_comparable() -> None:
    """THE POINT OF THIS FILE.

    Both arms hold 200 humans and 120 generations, so every count matches. They are drawn
    from DIFFERENT humans, so a measured gap would be partly a data gap and the writeup's
    central claim would not follow from the experiment. A count check passes this happily.
    """
    mirrors = [_generated(i, KIND_MIRROR) for i in range(120)]
    randoms = [_generated(i, KIND_UNMATCHED) for i in range(120)]
    arm_a = build_arm([_human(i) for i in range(200)], mirrors)
    arm_b = build_arm([_human(i) for i in range(1000, 1200)], randoms)
    assert len([s for s in arm_a if s.arm == SHARED]) == len(
        [s for s in arm_b if s.arm == SHARED]
    ), "the counts must match, or this test is not testing what it claims"
    with pytest.raises(IndexAssemblyError, match="same human and composite pool"):
        assert_arms_comparable(arm_a, arm_b)


def test_unequal_generated_budgets_are_refused() -> None:
    shared = _shared(100)
    arm_a = build_arm(shared, [_generated(i, KIND_MIRROR) for i in range(80)])
    arm_b = build_arm(shared, [_generated(i, KIND_UNMATCHED) for i in range(120)])
    with pytest.raises(IndexAssemblyError, match="unequal generated budgets"):
        assert_arms_comparable(arm_a, arm_b)


def test_two_empty_arms_are_refused_rather_than_called_equal() -> None:
    """0 == 0 passes an equality check and compares nothing."""
    shared = _shared(40)
    with pytest.raises(IndexAssemblyError, match="zero generated samples"):
        assert_arms_comparable(build_arm(shared, []), build_arm(shared, []))


def test_the_budget_cut_makes_unequal_pools_comparable() -> None:
    """The realistic case: the mirror arm rejects more, so it finishes smaller."""
    shared = _shared(200)
    mirrors = [_generated(i, KIND_MIRROR) for i in range(118)]
    randoms = [_generated(i, KIND_UNMATCHED) for i in range(377)]
    budget = min(len(mirrors), len(randoms))
    arm_a = build_arm(shared, mirrors, budget)
    arm_b = build_arm(shared, randoms, budget)
    assert_arms_comparable(arm_a, arm_b)
    assert ai_budget(arm_a) == ai_budget(arm_b) == 118


# -------------------------------------------------------------------------------- leakage


def test_a_group_split_across_two_splits_is_caught() -> None:
    """A photograph in train and its mirror in test inflates every test number."""
    photograph = ImageSample("img1", "grp1", "train", 0, KIND_HUMAN, "px/img1")
    mirror = ImageSample("mir1", "grp1", "test", 1, KIND_MIRROR, "px/mir1")
    with pytest.raises(IndexAssemblyError, match="two splits"):
        assert_no_group_leakage([photograph, mirror])


def test_a_correctly_grouped_index_passes() -> None:
    assert_no_group_leakage(_shared(120) + [_generated(i, KIND_MIRROR) for i in range(60)])


def test_a_single_class_split_is_refused() -> None:
    """AUROC is undefined and FPR is meaningless on one class."""
    with pytest.raises(IndexAssemblyError, match="both classes"):
        assert_both_classes([_human(0, "val"), _human(1, "val")], "val")


def test_a_two_class_split_passes() -> None:
    assert_both_classes([_human(0, "val"), _generated(1, KIND_MIRROR, "val")], "val")


# ------------------------------------------------------------------------------- the index


def test_the_index_round_trips(tmp_path) -> None:
    samples = _shared(60) + [_generated(i, KIND_MIRROR) for i in range(30)]
    path = tmp_path / "index.jsonl"
    write_index(samples, path)
    back = read_index(path)
    assert sorted(s.sample_id for s in back) == sorted(s.sample_id for s in samples)
    assert {s.mask_ref for s in back if s.kind == KIND_COMPOSITE_AI} == {
        s.mask_ref for s in samples if s.kind == KIND_COMPOSITE_AI
    }


def test_the_index_is_written_sorted(tmp_path) -> None:
    """So a diff between two runs means the corpus changed, not the filesystem's mood."""
    samples = _shared(40)
    path = tmp_path / "index.jsonl"
    write_index(list(reversed(samples)), path)
    ids = [json.loads(line)["sample_id"] for line in path.read_text().splitlines()]
    assert ids == sorted(ids)


def test_writing_a_leaking_index_raises_rather_than_writing_it(tmp_path) -> None:
    """The check must run before the file exists, or the bad dataset is on disk to be used."""
    leaking = [
        ImageSample("img1", "grp1", "train", 0, KIND_HUMAN, "px/img1"),
        ImageSample("mir1", "grp1", "test", 1, KIND_MIRROR, "px/mir1"),
    ]
    path = tmp_path / "index.jsonl"
    with pytest.raises(IndexAssemblyError):
        write_index(leaking, path)
    assert not path.exists()


def test_the_summary_reports_what_a_reader_needs() -> None:
    samples = _shared(120) + [_generated(i, KIND_MIRROR) for i in range(60)]
    summary = index_summary(samples)
    assert summary["ai_budget"] == 60
    assert summary["kinds"][KIND_COMPOSITE_AI] == 30
    assert summary["with_mask"] == 60
    assert set(summary["splits"]) == set(SPLITS)
    for split in SPLITS:
        assert set(summary["labels_per_split"][split]) == {0, 1}


# ------------------------------------------------------------------- the torch-dependent part

# NOT `pytest.importorskip` at module level. That call skips the ENTIRE module, so on a
# machine without torch the thirty index tests above would report as skipped rather than
# run, and the file would look green while checking nothing. That is the same failure shape
# this whole file is about, and it happened here on the first draft.
try:
    import torch
except ImportError:                                  # the index tests still run without it
    torch = None

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="needs torch; the index tests above do not and must still run",
)


@requires_torch
def test_localisation_is_scored_at_patch_resolution() -> None:
    """A head that predicts exactly the pasted region scores 1.0."""
    grid = 2
    mask = torch.zeros(8, 8)
    mask[:4, :] = 1.0                       # top half pasted
    logits = torch.tensor([[9.0, 9.0], [-9.0, -9.0]])
    assert patch_iou(logits, mask, grid) == pytest.approx(1.0)


@requires_torch
def test_a_head_that_points_at_the_wrong_half_scores_zero() -> None:
    grid = 2
    mask = torch.zeros(8, 8)
    mask[:4, :] = 1.0
    logits = torch.tensor([[-9.0, -9.0], [9.0, 9.0]])
    assert patch_iou(logits, mask, grid) == pytest.approx(0.0)


@requires_torch
def test_finding_nothing_in_an_empty_mask_is_a_success_not_a_division_by_zero() -> None:
    """A photograph has no AI region. Predicting none of it is correct, and 0/0 is not."""
    assert patch_iou(torch.full((2, 2), -9.0), torch.zeros(8, 8), 2) == pytest.approx(1.0)


@requires_torch
def test_iou_is_between_zero_and_one_on_a_partial_hit() -> None:
    grid = 2
    mask = torch.zeros(8, 8)
    mask[:4, :4] = 1.0                      # one quadrant
    logits = torch.tensor([[9.0, 9.0], [-9.0, -9.0]])   # two quadrants predicted
    assert patch_iou(logits, mask, grid) == pytest.approx(0.5)


def _png(colour: int, size: int = 32) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (colour, colour, colour)).save(buffer, format="PNG")
    return buffer.getvalue()


def _mask_png(size: int = 32) -> bytes:
    import io

    from PIL import Image

    img = Image.new("L", (size, size), 0)
    for y in range(size // 2):
        for x in range(size):
            img.putpixel((x, y), 255)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@requires_torch
def test_a_generated_whole_image_gets_an_all_ones_mask() -> None:
    """Not a placeholder. A fully generated image IS evidence everywhere, and saying so is
    what stops the local head from only ever seeing composites."""
    from forge.image.train import make_dataset

    sample = _generated(0, KIND_MIRROR)
    dataset = make_dataset([sample], lambda ref: _png(120), None, 16)
    assert dataset[0]["mask"].min().item() == pytest.approx(1.0)
    assert dataset[0]["label"].item() == pytest.approx(1.0)


@requires_torch
def test_a_photograph_gets_an_all_zeros_mask() -> None:
    from forge.image.train import make_dataset

    dataset = make_dataset([_human(0)], lambda ref: _png(120), None, 16)
    assert dataset[0]["mask"].max().item() == pytest.approx(0.0)


@requires_torch
def test_a_composite_reads_its_stored_mask_instead() -> None:
    """The regression this guards: falling back to the constant mask for a composite would
    label the whole image as evidence and destroy the only real localisation signal."""
    from forge.image.train import make_dataset

    dataset = make_dataset(
        [_composite(0, KIND_COMPOSITE_AI)], lambda ref: _png(120), lambda ref: _mask_png(), 16
    )
    mask = dataset[0]["mask"]
    assert mask[:8, :].mean().item() == pytest.approx(1.0)
    assert mask[8:, :].mean().item() == pytest.approx(0.0)


@requires_torch
def test_pixels_are_scaled_to_unit_range() -> None:
    from forge.image.train import make_dataset

    dataset = make_dataset([_human(0)], lambda ref: _png(255), None, 16)
    pixels = dataset[0]["pixel_values"]
    assert pixels.shape == (3, 16, 16)
    assert 0.0 <= pixels.min().item() and pixels.max().item() <= 1.0


@requires_torch
def test_the_summary_reports_the_threshold_it_was_given() -> None:
    import numpy as np

    from forge.image.train import summarise

    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    result = summarise(probabilities, labels, [0.5, 0.7], threshold=0.5)
    assert result.threshold == pytest.approx(0.5)
    assert result.auroc == pytest.approx(1.0)
    assert result.n_human == 2 and result.n_ai == 2
    assert result.localisation_iou == pytest.approx(0.6)


@requires_torch
def test_a_split_with_no_composites_reports_no_localisation_rather_than_zero() -> None:
    """Reporting 0.0 would read as a model that localises nothing, which is a different claim."""
    import math

    import numpy as np

    from forge.image.train import summarise

    result = summarise(np.array([0.1, 0.9]), np.array([0, 1]), [], threshold=0.5)
    assert math.isnan(result.localisation_iou)
    assert result.n_localised == 0


@requires_torch
def test_a_resized_mask_stays_binary() -> None:
    """THE BUG THIS CAME FROM.

    make_dataset resized masks with PIL's default filter. On a hard-edged mask a smooth
    kernel puts a ramp along every boundary, and a bicubic kernel OVERSHOOTS, so values leave
    [0, 1]. Nothing raises. The local head is then trained against a soft, slightly
    out-of-range target that no composite ever had, and the IoU it is scored on quietly
    disagrees with the mask that was stored.
    """
    from forge.image.train import make_dataset

    dataset = make_dataset(
        [_composite(0, KIND_COMPOSITE_AI)], lambda ref: _png(120), lambda ref: _mask_png(), 15
    )
    mask = dataset[0]["mask"]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}, (
        f"resampling produced intermediate mask values: {sorted(mask.unique().tolist())[:5]}"
    )
    assert 0.0 <= mask.min().item() and mask.max().item() <= 1.0
