"""Limiting a run must sample the corpus, not truncate it.

THE BUG THIS CAME FROM. read_humans stopped reading as soon as it had `limit`
documents. The human lake is partitioned by source and split, and the files are
read in sorted order, so a limited run drew every document from whichever
partition sorts first. A 200-document check produced 200 test-split documents
and zero train, which looked like a quirk of a smoke run.

It stopped being a quirk the moment the real experiment was cut from 60k to 30k
documents per arm to fit the compute budget: that run would have been drawn from
one corner of the corpus, with the wrong split proportions and the wrong source
mix, and no error would have been raised. The dataset card would have described
a corpus that was never built.
"""

from __future__ import annotations

from collections import Counter

import pytest

from forge.common.schemas import Split
from forge.generation.run import HumanRef, subsample

SOURCES = ("fw", "fwe")
SPLITS = (Split.TRAIN, Split.VAL, Split.TEST)


def _corpus(n: int = 3000) -> list[HumanRef]:
    """A corpus laid out the way the parquet files are: grouped, not interleaved."""
    refs: list[HumanRef] = []
    for source in SOURCES:
        for split in SPLITS:
            for i in range(n // (len(SOURCES) * len(SPLITS))):
                doc = f"{source}_{split.value}_{i:05d}"
                refs.append(HumanRef(doc, f"grp_{doc}", "text " * 60, source, split))
    return refs


def test_returns_exactly_the_requested_count() -> None:
    assert len(subsample(_corpus(), 300)) == 300


def test_draws_from_every_split(  ) -> None:
    """The regression. Truncation returned one split; sampling must return all three."""
    got = Counter(r.split for r in subsample(_corpus(), 300))
    assert set(got) == set(SPLITS), f"only these splits were sampled: {dict(got)}"


def test_draws_from_every_source() -> None:
    got = Counter(r.domain for r in subsample(_corpus(), 300))
    assert set(got) == set(SOURCES), f"only these sources were sampled: {dict(got)}"


def test_split_proportions_are_roughly_preserved() -> None:
    """A uniform hash over doc ids should track the corpus mix within sampling noise."""
    corpus = _corpus()
    want = Counter(r.split for r in corpus)
    got = Counter(r.split for r in subsample(corpus, 600))
    for split in SPLITS:
        expected = 600 * want[split] / len(corpus)
        assert abs(got[split] - expected) < 0.25 * expected, (
            f"{split} was sampled {got[split]} times, expected about {expected:.0f}"
        )


def test_selection_is_deterministic() -> None:
    """Two runs of the same config must build the same dataset."""
    corpus = _corpus()
    first = [r.doc_id for r in subsample(corpus, 250)]
    second = [r.doc_id for r in subsample(corpus, 250)]
    assert first == second


def test_selection_does_not_depend_on_input_order() -> None:
    """File ordering must not change which documents a limit selects."""
    corpus = _corpus()
    shuffled = corpus[len(corpus) // 2 :] + corpus[: len(corpus) // 2]
    assert sorted(r.doc_id for r in subsample(corpus, 250)) == sorted(
        r.doc_id for r in subsample(shuffled, 250)
    )


def test_smaller_limits_nest_inside_larger_ones() -> None:
    """A probe must be a genuine preview of the run that follows it.

    Without nesting, a 2k probe and a 30k run would share almost no documents and the
    probe's rejection statistics would say little about the real run.
    """
    corpus = _corpus()
    small = {r.doc_id for r in subsample(corpus, 200)}
    large = {r.doc_id for r in subsample(corpus, 800)}
    assert small <= large


def test_output_keeps_corpus_order() -> None:
    """Selection must not reorder what it returns, or output order depends on the limit."""
    corpus = _corpus()
    picked = subsample(corpus, 300)
    order = {r.doc_id: i for i, r in enumerate(corpus)}
    assert [order[r.doc_id] for r in picked] == sorted(order[r.doc_id] for r in picked)


@pytest.mark.parametrize("limit", [1, 7, 999])
def test_various_limits_are_exact(limit: int) -> None:
    assert len(subsample(_corpus(), limit)) == limit
