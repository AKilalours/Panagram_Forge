"""Pinning a generator roster must be able to tell a write from a no-op.

THE BUG THIS CAME FROM. `forge pin-revisions --write` resolved all six open-source
families, printed "pinned 6 revisions", and left one of them unpinned. The
substitution was a literal str.replace of

    "model_id: <id>\\n    revision: TODO_PIN_AT_FIRST_RUN"

which requires a newline directly after the model id. Two entries in the roster
carry a trailing comment on that line, so for those the literal never matched.
str.replace signals a no-op by returning the string unchanged, and the success
count was len(resolved), meaning HuggingFace lookups that succeeded, not lines
rewritten. So the reported number could never disagree with reality, because it
was not measuring reality.

Consequence had it gone unnoticed: generation would run against whatever the
unpinned repo's default branch pointed at that day, while the config claimed a
fixed roster. The writeup's reproducibility claim would be false for exactly one
generator, and nothing would crash.
"""

from __future__ import annotations

import pytest

from forge.generation.pinning import TODO, pin_revision, unpinned_lines

SHA = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

PLAIN = """families:
  - family: qwen
    model_id: Qwen/Qwen2.5-3B-Instruct
    revision: TODO_PIN_AT_FIRST_RUN
    role: held_in
"""

# The exact shape that defeated the old implementation.
WITH_MODEL_ID_COMMENT = """families:
  - family: gemma
    model_id: google/gemma-2-2b-it        # GATED. Non-gated fallback: Qwen/Qwen2.5-1.5B-Instruct
    revision: TODO_PIN_AT_FIRST_RUN
    role: held_out
"""

WITH_REVISION_COMMENT = """families:
  - family: qwen
    model_id: Qwen/Qwen2.5-3B-Instruct
    revision: TODO_PIN_AT_FIRST_RUN     # `forge pin-revisions` fills these in
    role: held_in
"""


def test_pins_a_plain_entry() -> None:
    out, n = pin_revision(PLAIN, "Qwen/Qwen2.5-3B-Instruct", SHA)
    assert n == 1
    assert f"revision: {SHA}" in out
    assert TODO not in out


def test_pins_an_entry_whose_model_id_line_has_a_trailing_comment() -> None:
    """The regression. This is the case that silently did nothing."""
    out, n = pin_revision(WITH_MODEL_ID_COMMENT, "google/gemma-2-2b-it", SHA)
    assert n == 1, "a trailing comment on the model_id line must not defeat the match"
    assert f"revision: {SHA}" in out
    assert TODO not in out


def test_preserves_the_comment_it_matched_past() -> None:
    """Rewriting must not eat the explanatory comment; it is load-bearing documentation."""
    out, _ = pin_revision(WITH_MODEL_ID_COMMENT, "google/gemma-2-2b-it", SHA)
    assert "# GATED. Non-gated fallback: Qwen/Qwen2.5-1.5B-Instruct" in out


def test_a_comment_after_the_revision_value_is_replaced_not_kept() -> None:
    """The scaffolding hint stops being true once the value is a sha, so it goes."""
    out, n = pin_revision(WITH_REVISION_COMMENT, "Qwen/Qwen2.5-3B-Instruct", SHA)
    assert n == 1
    assert f"revision: {SHA}" in out
    assert "fills these in" not in out


def test_reports_zero_when_the_model_is_absent() -> None:
    """A caller must be able to detect the no-op, which str.replace could not express."""
    out, n = pin_revision(PLAIN, "some/model-not-in-the-roster", SHA)
    assert n == 0
    assert out == PLAIN


def test_is_idempotent_and_can_repin() -> None:
    """Pinning an already-pinned roster must update it, not refuse it.

    The old literal only matched the TODO placeholder, so a roster could be pinned
    exactly once and never refreshed.
    """
    once, n1 = pin_revision(PLAIN, "Qwen/Qwen2.5-3B-Instruct", SHA)
    newer = "b" * 40
    twice, n2 = pin_revision(once, "Qwen/Qwen2.5-3B-Instruct", newer)
    assert (n1, n2) == (1, 1)
    assert f"revision: {newer}" in twice
    assert SHA not in twice


def test_does_not_touch_a_different_family() -> None:
    two = PLAIN + """  - family: phi
    model_id: microsoft/Phi-3.5-mini-instruct
    revision: TODO_PIN_AT_FIRST_RUN
    role: held_in
"""
    out, n = pin_revision(two, "Qwen/Qwen2.5-3B-Instruct", SHA)
    assert n == 1
    assert out.count(TODO) == 1, "only the named family may be rewritten"


def test_does_not_match_across_an_intervening_line() -> None:
    """The revision line must be the one directly under the model id.

    Without this the pattern could bind a family's model id to a neighbouring
    family's revision line and pin the wrong entry.
    """
    interleaved = """families:
  - family: qwen
    model_id: Qwen/Qwen2.5-3B-Instruct
    released: 2024-09
    revision: TODO_PIN_AT_FIRST_RUN
"""
    _, n = pin_revision(interleaved, "Qwen/Qwen2.5-3B-Instruct", SHA)
    assert n == 0


@pytest.mark.parametrize(
    "text,expected",
    [(PLAIN, 1), (WITH_MODEL_ID_COMMENT, 1), (PLAIN + WITH_MODEL_ID_COMMENT, 2)],
)
def test_unpinned_lines_counts_the_placeholders(text: str, expected: int) -> None:
    assert len(unpinned_lines(text)) == expected


def test_unpinned_lines_is_empty_once_pinned() -> None:
    out, _ = pin_revision(PLAIN, "Qwen/Qwen2.5-3B-Instruct", SHA)
    assert unpinned_lines(out) == []
