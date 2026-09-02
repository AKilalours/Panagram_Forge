"""A limited load must sample every source, not just the first one read.

THE BUG THIS CAME FROM. load_examples appended humans, then AI, then mixed
documents to one list, returning early once it held `limit` rows:

    if limit and len(out) >= limit:
        return out

Humans are read first, so any limited load returned humans only. `--smoke`,
the 200-example check that exists precisely to catch config errors before a
paid training run, therefore trained on 200 human documents and zero AI
documents. It completed, printed a loss, and proved nothing: a binary
classifier shown one class cannot fail in a way that check would notice.

Same defect as the corpus loader's prefix truncation, in a different file. An
early return inside a source-specific loop is never a subsample; it is a filter
on source.
"""

from __future__ import annotations

from collections import Counter

import pytest

from forge.training.data import RawExample, _by_source_and_split, interleave


def _rows(prefix: str, label: int, n: int, split: str = "train") -> list[RawExample]:
    return [
        RawExample(f"{prefix}_{split}_{i}", f"grp_{prefix}{i}", split, "text", label, None)
        for i in range(n)
    ]


def _partitioned(prefix: str, label: int) -> list[RawExample]:
    """Laid out the way the parquet files are: all of one split, then the next."""
    return (
        _rows(prefix, label, 100, "test")
        + _rows(prefix, label, 800, "train")
        + _rows(prefix, label, 100, "val")
    )


HUMANS = _rows("h", 0, 500)
AI = _rows("a", 1, 500)
MIXED = _rows("m", 1, 100)


def test_unlimited_load_returns_everything_in_order() -> None:
    got = interleave(HUMANS, AI, MIXED, limit=None)
    assert len(got) == 1100
    # Compare against the buckets themselves rather than literal ids, so a change to the
    # fixture's naming cannot leave this assertion silently checking the wrong thing.
    assert got[0] is HUMANS[0]
    assert got[-1] is MIXED[-1]
    assert [r.doc_id for r in got] == [r.doc_id for r in HUMANS + AI + MIXED]


def test_a_limited_load_contains_both_classes() -> None:
    """The regression. This is exactly what --smoke did."""
    got = interleave(HUMANS, AI, MIXED, limit=200)
    labels = Counter(r.label for r in got)
    assert labels[0] > 0, "no human documents in a limited load"
    assert labels[1] > 0, (
        "no AI documents in a limited load; a classifier trained on this cannot fail "
        "in a way a smoke test would notice"
    )


def test_a_limited_load_draws_from_every_source() -> None:
    got = interleave(HUMANS, AI, MIXED, limit=300)
    prefixes = {r.doc_id[0] for r in got}
    assert prefixes == {"h", "a", "m"}, f"only these sources appeared: {prefixes}"


def test_classes_are_roughly_balanced_in_a_limited_load() -> None:
    """Round-robin over two equal buckets should land near 50/50."""
    got = interleave(HUMANS, AI, limit=200)
    labels = Counter(r.label for r in got)
    assert abs(labels[0] - labels[1]) <= 2


@pytest.mark.parametrize("limit", [1, 2, 7, 199, 1100])
def test_the_limit_is_exact(limit: int) -> None:
    assert len(interleave(HUMANS, AI, MIXED, limit=limit)) == limit


def test_a_limit_larger_than_the_data_returns_everything() -> None:
    assert len(interleave(HUMANS, AI, MIXED, limit=99999)) == 1100


def test_an_empty_source_does_not_stall_the_round_robin() -> None:
    """A missing AI directory must not produce an infinite loop or a short read."""
    got = interleave(HUMANS, [], limit=50)
    assert len(got) == 50
    assert all(r.label == 0 for r in got)


def test_selection_is_deterministic() -> None:
    first = [r.doc_id for r in interleave(HUMANS, AI, MIXED, limit=137)]
    second = [r.doc_id for r in interleave(HUMANS, AI, MIXED, limit=137)]
    assert first == second


def test_no_document_is_returned_twice() -> None:
    got = interleave(HUMANS, AI, MIXED, limit=400)
    assert len({r.doc_id for r in got}) == len(got)


def test_a_limited_load_spans_every_split() -> None:
    """The second half of the bug.

    Round-robining over sources alone still drew from the front of each list, and the
    parquet files are partitioned by split, so a 200-row limit returned only `test`
    rows. Training then failed with "no training windows after windowing", which names
    the symptom rather than the cause.
    """
    buckets = _by_source_and_split(_partitioned("h", 0), _partitioned("a", 1))
    got = interleave(*buckets, limit=200)
    splits = Counter(r.split for r in got)
    assert set(splits) == {"train", "val", "test"}, f"only these splits appeared: {dict(splits)}"


def test_a_limited_load_spans_splits_and_classes_together() -> None:
    buckets = _by_source_and_split(_partitioned("h", 0), _partitioned("a", 1))
    got = interleave(*buckets, limit=120)
    pairs = {(r.split, r.label) for r in got}
    for split in ("train", "val", "test"):
        assert (split, 0) in pairs, f"no human rows in {split}"
        assert (split, 1) in pairs, f"no AI rows in {split}"


def test_bucketing_preserves_every_row() -> None:
    sources = [_partitioned("h", 0), _partitioned("a", 1)]
    buckets = _by_source_and_split(*sources)
    assert sum(len(b) for b in buckets) == sum(len(s) for s in sources)
    assert len({r.doc_id for b in buckets for r in b}) == 2000
