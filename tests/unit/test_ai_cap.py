"""The AI budget must be held equal across arms, whatever generation yielded.

WHY THIS EXISTS. Each arm's validator rejects at its own rate. The mirror arm
accepts a length ratio in [0.6, 1.6] and rejected about 40% of first-pass
generations; the random arm accepts [0.5, 2.0] and rejects far less. So the two
arms finish generation with different document counts even though both were
asked for 30,000.

Training on those counts as-is would confound the experiment it exists to run.
Arm A winning could simply mean arm A had thousands more documents, and arm B
winning despite fewer would be a different, weaker claim than the one the
project set out to test. The plan's own words: the budget is held equal so that
any improvement is attributable to matching rather than to volume.

cap_documents restores that, deterministically and without regenerating
anything.
"""

from __future__ import annotations

from collections import Counter

import pytest

from forge.training.data import RawExample, cap_documents


def _rows(n: int, split: str, prefix: str = "d") -> list[RawExample]:
    return [
        RawExample(f"{prefix}_{split}_{i}", f"grp_{prefix}_{i}", split, "text", 1, None)
        for i in range(n)
    ]


CORPUS = _rows(800, "train") + _rows(100, "val") + _rows(100, "test")


def test_caps_to_exactly_the_requested_count() -> None:
    assert len(cap_documents(CORPUS, 500)) == 500


@pytest.mark.parametrize("cap", [1, 7, 333, 999, 1000])
def test_various_caps_are_exact(cap: int) -> None:
    assert len(cap_documents(CORPUS, cap)) == cap


def test_split_proportions_survive_the_cap() -> None:
    """A global cap would let one split absorb the whole reduction."""
    got = Counter(r.split for r in cap_documents(CORPUS, 500))
    assert abs(got["train"] - 400) <= 2
    assert abs(got["val"] - 50) <= 2
    assert abs(got["test"] - 50) <= 2


def test_no_split_is_emptied() -> None:
    got = Counter(r.split for r in cap_documents(CORPUS, 100))
    assert set(got) == {"train", "val", "test"}


def test_selection_is_deterministic() -> None:
    """Two arms capped to the same number must be capped the same way."""
    first = [r.doc_id for r in cap_documents(CORPUS, 400)]
    second = [r.doc_id for r in cap_documents(CORPUS, 400)]
    assert first == second


def test_selection_does_not_depend_on_read_order() -> None:
    shuffled = CORPUS[500:] + CORPUS[:500]
    assert sorted(r.doc_id for r in cap_documents(CORPUS, 400)) == sorted(
        r.doc_id for r in cap_documents(shuffled, 400)
    )


def test_output_keeps_input_order() -> None:
    picked = cap_documents(CORPUS, 300)
    order = {r.doc_id: i for i, r in enumerate(CORPUS)}
    assert [order[r.doc_id] for r in picked] == sorted(order[r.doc_id] for r in picked)


def test_smaller_caps_nest_inside_larger_ones() -> None:
    """Two arms capped to different numbers should still overlap sensibly."""
    small = {r.doc_id for r in cap_documents(CORPUS, 200)}
    large = {r.doc_id for r in cap_documents(CORPUS, 600)}
    # Per-split ranking is stable, so the smaller selection sits inside the larger one.
    assert len(small - large) == 0


def test_two_arms_of_different_sizes_end_up_equal() -> None:
    """The scenario this was written for."""
    arm_a = _rows(2700, "train", "a") + _rows(300, "val", "a")
    arm_b = _rows(2200, "train", "b") + _rows(250, "val", "b")
    cap = min(len(arm_a), len(arm_b))
    assert len(cap_documents(arm_a, cap)) == len(cap_documents(arm_b, cap)) == cap


def test_a_cap_at_or_above_the_corpus_is_a_no_op() -> None:
    assert len(cap_documents(CORPUS, 1000)) == 1000
