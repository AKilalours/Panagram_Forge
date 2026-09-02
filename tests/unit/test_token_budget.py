"""Generation length must be per document, not one number for the whole corpus.

THE BUG THIS CAME FROM, and it is the most expensive one in the project so far.

Every document in both arms was generated with a single max_new_tokens of 640, read from
the decoding grid. Both arms already computed a correct per-document length target and put
it in the prompt: the mirror arm from its source document, the control arm sampled from the
human corpus's own length distribution. The models ignored the instruction and wrote to the
cap. Measured on the finished corpora:

    human    median 250 words
    mirror   median 364 words   AUROC from length alone 0.774
    random   median 394 words   AUROC from length alone 0.841

A detector scores 0.84 on the control arm without reading a word. Both arms would have
trained a length detector, and the mirror-versus-random result would have been a comparison
between two of them.

Nothing errored. The prompts were correct, the validator was correct, the counts looked
healthy, and the stated length target was simply advice that nothing enforced. That is the
same shape as every other bug in this project: a mechanism that reports success while the
thing it describes is not happening.

WHY THE HEADROOM IS NOT ZERO. Setting the budget exactly at the target would cut generations
off mid-clause, and text that stops mid-word is far more detectable than text that is long.
Trading a length shortcut for a truncation shortcut is not a fix.
"""

from __future__ import annotations

import pytest

from forge.generation.assignment import (
    LENGTH_HEADROOM,
    MIN_NEW_TOKENS,
    TOKENS_PER_WORD,
    assign_decoding,
    token_budget,
)

GRID = {"temperature": [0.7, 1.0], "top_p": [0.9], "max_new_tokens": 640}


def test_the_budget_tracks_the_target() -> None:
    """The regression: a 200-word source and a 500-word source must not get the same budget."""
    assert token_budget(200, 640) < token_budget(500, 640)


def test_a_short_source_gets_a_short_budget() -> None:
    """The specific failure. Under the old code this was 640 for every document."""
    assert token_budget(180, 640) < 640


@pytest.mark.parametrize("words", [150, 250, 400])
def test_the_budget_leaves_room_to_finish_a_sentence(words: int) -> None:
    """Enough headroom that the model ends cleanly, or truncation becomes the new shortcut."""
    assert token_budget(words, 4096) > words * TOKENS_PER_WORD


def test_the_budget_stays_inside_the_validator_window(  ) -> None:
    """Headroom must not be so generous that it recreates the overshoot it was fixing.

    The mirror validator accepts a word-count ratio in [0.6, 1.6]. If a model runs to the
    cap, the resulting ratio is roughly LENGTH_HEADROOM, so the headroom is what the ratio
    distribution centres on. A headroom at or above 1.6 would put the centre of the
    distribution on the rejection boundary.
    """
    assert 1.0 < LENGTH_HEADROOM < 1.5


def test_a_tiny_source_still_gets_a_usable_budget() -> None:
    """A 10-word target must not produce a budget too small to write anything."""
    assert token_budget(10, 640) == MIN_NEW_TOKENS


def test_the_grid_maximum_is_still_a_ceiling() -> None:
    """A very long source must not be allowed to blow past the configured context."""
    assert token_budget(100_000, 640) == 640


def test_a_non_positive_target_is_refused() -> None:
    """Silently substituting a default would put the global cap back for those documents."""
    with pytest.raises(ValueError):
        token_budget(0, 640)


def test_decoding_carries_the_per_document_budget() -> None:
    short = assign_decoding("doc-a", GRID, target_words=160)
    long = assign_decoding("doc-b", GRID, target_words=520)
    assert short.max_new_tokens < long.max_new_tokens


def test_two_documents_with_the_same_target_get_the_same_budget() -> None:
    """The budget is a function of the target, not of the id."""
    a = assign_decoding("doc-a", GRID, target_words=300)
    b = assign_decoding("doc-b", GRID, target_words=300)
    assert a.max_new_tokens == b.max_new_tokens


def test_adding_a_length_target_does_not_change_which_decoding_is_picked() -> None:
    """THE COMPATIBILITY ASSERTION.

    Temperature, top_p and seed come from hashes of the document id. If adding the length
    budget perturbed those, the regenerated corpus would differ from the old one in two
    ways at once and the length fix could not be isolated as the cause of any change.
    """
    for doc in ("doc-a", "doc-b", "doc-c", "doc-d"):
        without = assign_decoding(doc, GRID)
        with_target = assign_decoding(doc, GRID, target_words=300)
        assert (without.temperature, without.top_p, without.seed) == (
            with_target.temperature,
            with_target.top_p,
            with_target.seed,
        )


def test_omitting_the_target_keeps_the_old_global_behaviour() -> None:
    """Callers without a length target still work, and get the grid's value."""
    assert assign_decoding("doc-a", GRID).max_new_tokens == 640


def test_the_budget_is_deterministic() -> None:
    assert token_budget(275, 640) == token_budget(275, 640)


def test_a_realistic_corpus_target_lands_well_under_the_old_cap() -> None:
    """The human median is 250 words. That document used to get 640 tokens."""
    assert token_budget(250, 640) < 500
