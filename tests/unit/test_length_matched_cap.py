"""Capping the arms to an equal COUNT is not enough; they must match in shape.

THE CONFOUND. The mirror arm's validator rejects a generation whose length ratio to its
source falls outside [0.6, 1.6]. Generation used one max_new_tokens for every document, so
a short source could never be matched: the model wrote to the cap and overshot. Length
overshoot was 85% of all rejections in the first real run, and the mirrors that survived
skew long.

The control arm draws its target lengths from the whole human corpus, so it does not skew.
Cap both arms to the same number and they still differ in length distribution, which a
detector can learn. A win for either arm would then have an obvious alternative
explanation, and the experiment would not answer the question it was built to ask.
"""

from __future__ import annotations

from collections import Counter

import pytest

from forge.training.data import LENGTH_BINS, RawExample, cap_documents_matching, length_bin


def _row(doc_id: str, words: int, split: str = "train") -> RawExample:
    return RawExample(doc_id, f"grp_{doc_id}", split, "word " * words, 1, None)


def _corpus(prefix: str, spec: dict[int, int], split: str = "train") -> list[RawExample]:
    """Build rows with a chosen length profile: {word_count: how_many}."""
    rows = []
    for words, count in spec.items():
        rows.extend(_row(f"{prefix}_{words}_{i}", words, split) for i in range(count))
    return rows


# Reference skews long, the way accepted mirrors do.
REFERENCE = _corpus("m", {100: 50, 200: 100, 300: 200, 600: 400, 900: 250})
# Candidate is flat, the way the control arm is.
CANDIDATE = _corpus("r", {100: 200, 200: 200, 300: 200, 600: 200, 900: 200})


def _profile(rows: list[RawExample]) -> dict[int, float]:
    counts = Counter(length_bin(r.text) for r in rows)
    total = sum(counts.values())
    return {b: counts[b] / total for b in counts}


def test_bins_are_ordered_and_cover_short_and_long() -> None:
    assert list(LENGTH_BINS) == sorted(LENGTH_BINS)
    assert length_bin("word " * 10) == 0
    assert length_bin("word " * 5000) == len(LENGTH_BINS) - 1


def test_the_cap_is_exact() -> None:
    assert len(cap_documents_matching(CANDIDATE, REFERENCE, 400)) == 400


@pytest.mark.parametrize("cap", [1, 37, 500, 999])
def test_various_caps_are_exact(cap: int) -> None:
    assert len(cap_documents_matching(CANDIDATE, REFERENCE, cap)) == cap


def test_the_result_tracks_the_reference_distribution() -> None:
    """The regression. A plain count cap would leave the candidate's flat profile."""
    capped = cap_documents_matching(CANDIDATE, REFERENCE, 500)
    want, got = _profile(REFERENCE), _profile(capped)
    for bucket, fraction in want.items():
        assert abs(got.get(bucket, 0.0) - fraction) < 0.08, (
            f"bin {bucket}: reference {fraction:.2f}, capped {got.get(bucket, 0):.2f}"
        )


def test_matching_beats_an_unmatched_cap() -> None:
    """Make the improvement explicit rather than assumed."""
    from forge.training.data import cap_documents

    plain = _profile(cap_documents(CANDIDATE, 500))
    matched = _profile(cap_documents_matching(CANDIDATE, REFERENCE, 500))
    want = _profile(REFERENCE)
    error = lambda p: sum(abs(p.get(b, 0.0) - f) for b, f in want.items())  # noqa: E731
    assert error(matched) < error(plain)


def test_selection_is_deterministic() -> None:
    first = [r.doc_id for r in cap_documents_matching(CANDIDATE, REFERENCE, 300)]
    second = [r.doc_id for r in cap_documents_matching(CANDIDATE, REFERENCE, 300)]
    assert first == second


def test_selection_does_not_depend_on_read_order() -> None:
    shuffled = CANDIDATE[600:] + CANDIDATE[:600]
    assert sorted(r.doc_id for r in cap_documents_matching(CANDIDATE, REFERENCE, 300)) == sorted(
        r.doc_id for r in cap_documents_matching(shuffled, REFERENCE, 300)
    )


def test_output_keeps_input_order() -> None:
    picked = cap_documents_matching(CANDIDATE, REFERENCE, 300)
    order = {r.doc_id: i for i, r in enumerate(CANDIDATE)}
    assert [order[r.doc_id] for r in picked] == sorted(order[r.doc_id] for r in picked)


def test_splits_are_matched_as_well_as_lengths() -> None:
    """A cell is (split, length bin), so a cap cannot quietly drain one split."""
    reference = REFERENCE + _corpus("mv", {300: 100, 600: 100}, split="val")
    candidate = CANDIDATE + _corpus("rv", {300: 150, 600: 150}, split="val")
    capped = cap_documents_matching(candidate, reference, 600)
    got = Counter(r.split for r in capped)
    assert got["val"] > 0 and got["train"] > 0


def test_a_short_cell_does_not_shrink_the_result() -> None:
    """If one bin cannot supply its share, the rest make it up; the count stays exact.

    Otherwise the two arms would differ in size, which is the very thing the cap exists to
    prevent.
    """
    scarce = _corpus("s", {100: 5, 600: 400})
    capped = cap_documents_matching(scarce, REFERENCE, 300)
    assert len(capped) == 300


def test_a_cap_at_or_above_the_candidate_is_a_no_op() -> None:
    assert len(cap_documents_matching(CANDIDATE, REFERENCE, 99_999)) == len(CANDIDATE)
