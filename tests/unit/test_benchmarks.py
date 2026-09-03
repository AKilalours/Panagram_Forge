"""External benchmark loaders and contamination.

Fixtures match the schemas recorded in data_spec_v1 section 1.1, read from the live
dataset cards. The real datasets are unreachable from the development environment, so
these tests verify parsing against the recorded schema, not the download.
"""

import pytest

from forge.evaluation.benchmarks import (
    LabelPolarityError,
    SchemaMismatch,
    assert_mage_polarity,
    group_aware_scores,
    parse_hc3,
    parse_mage,
    parse_raid,
)
from forge.evaluation.contamination import canonical_hash, check


def _raid_row(**kw):
    base = dict(
        id="r1", adv_source_id=None, source_id="s1", model="gpt-4", decoding="sampling",
        repetition_penalty="no", attack="none", domain="news", title="t", prompt="p",
        generation="Some generated news text of reasonable length for the purpose.",
    )
    base.update(kw)
    return base


# ------------------------------------------------------------------------ RAID

def test_raid_human_rows_are_label_zero():
    docs = parse_raid([_raid_row(model="human"), _raid_row(id="r2", model="gpt-4")])
    assert docs[0].label == 0 and docs[1].label == 1


def test_raid_excludes_code_and_non_english_domains():
    """FORGE v1 is not a code or multilingual detector. Scoring it on those moves the
    number for reasons unrelated to the research question."""
    rows = [_raid_row(id=f"r{i}", domain=d) for i, d in enumerate(["news", "code", "czech", "german", "poetry"])]
    kept = {d.domain for d in parse_raid(rows)}
    assert kept == {"news", "poetry"}


def test_raid_attacked_variants_share_the_base_group():
    """The 12 attacks are applied to the same base generations. Scoring them
    independently counts one base text up to thirteen times."""
    rows = [_raid_row(id=f"r{i}", attack=a, source_id="base1")
            for i, a in enumerate(["none", "homoglyph", "whitespace", "paraphrase"])]
    docs = parse_raid(rows)
    assert len({d.group_id for d in docs}) == 1


def test_group_aware_scoring_collapses_variants():
    rows = [_raid_row(id=f"r{i}", attack=a, source_id="base1")
            for i, a in enumerate(["none", "homoglyph", "whitespace", "paraphrase"])]
    docs = parse_raid(rows)
    y, s = group_aware_scores(docs, [1.0, 0.0, 1.0, 0.0])
    assert len(y) == 1 and abs(s[0] - 0.5) < 1e-9


def test_raid_can_exclude_attacks_for_a_clean_baseline():
    rows = [_raid_row(id="a", attack="none"), _raid_row(id="b", attack="homoglyph")]
    assert len(parse_raid(rows, include_attacks=False)) == 1


def test_a_changed_upstream_schema_fails_loudly():
    with pytest.raises(SchemaMismatch):
        parse_raid([{"id": "r1", "text": "oops, renamed column"}])


# ------------------------------------------------------------------------ MAGE

def test_mage_label_zero_is_machine():
    """THIS TEST USED TO ASSERT THE OPPOSITE, AND THAT IS WHY NOTHING CAUGHT THE BUG.

    It read `test_mage_machine_label_maps_to_one` and locked in the guess that MAGE's
    label 1 means machine. The guess was wrong, and the test made it permanent: every run
    was green while the parser inverted every MAGE label.

    A test written from an assumption does not verify the assumption. It preserves it, and
    it converts a question nobody checked into a fact nobody can question. The fixture even
    used src values "gpt" and "human", which look like evidence and were invented here
    rather than read from the dataset.

    The convention is now MEASURED over 6,000 real rows: every label-0 src names a
    generator, every label-1 src names a human corpus, 100% against 0%. See
    scripts/mage_polarity_probe.py.
    """
    docs = parse_mage([
        {"text": "a", "label": 0, "src": "eli5_machine_continuation_flan_t5_base"},
        {"text": "b", "label": 1, "src": "xsum_human"},
    ])
    assert docs[0].label == 1, "MAGE label 0 is machine-generated"
    assert docs[1].label == 0, "MAGE label 1 is human"


def test_mage_convention_can_still_be_overridden_explicitly():
    """The override stays, because a dataset can be republished with a new convention.

    What changed is the DEFAULT, and that the default is now a measurement rather than a
    guess.
    """
    docs = parse_mage([{"text": "a", "label": 1, "src": "gpt"}], machine_label=1)
    assert docs[0].label == 1


def test_polarity_check_catches_inverted_labels():
    """An inverted label turns an AUROC of 0.05 into a reported 0.95 and nothing crashes."""
    docs = parse_mage([{"text": f"t{i}", "label": i % 2, "src": "s"} for i in range(20)])
    good = [0.9 if d.label == 1 else 0.1 for d in docs]
    assert_mage_polarity(docs, good)  # must not raise
    with pytest.raises(LabelPolarityError):
        assert_mage_polarity(docs, [1 - s for s in good])


def test_polarity_check_also_catches_a_near_random_detector():
    docs = parse_mage([{"text": f"t{i}", "label": i % 2, "src": "s"} for i in range(20)])
    with pytest.raises(LabelPolarityError):
        assert_mage_polarity(docs, [0.5] * 20)


# ------------------------------------------------------------------------ HC3

def _hc3_row(qid="1"):
    return {
        "id": qid, "question": "why is the sky blue?",
        "human_answers": ["Rayleigh scattering, roughly.", "Short wavelengths scatter more."],
        "chatgpt_answers": ["The sky appears blue due to Rayleigh scattering of sunlight."],
        "source": "open_qa",
    }


def test_hc3_rows_expand_to_several_documents():
    """Answers are LISTS. Row counts are not document counts."""
    docs = parse_hc3([_hc3_row()])
    assert len(docs) == 3
    assert sum(d.label for d in docs) == 1


def test_every_answer_from_one_question_shares_a_group():
    """Otherwise the same question lands on both sides of any split or resample."""
    docs = parse_hc3([_hc3_row()])
    assert len({d.group_id for d in docs}) == 1


def test_hc3_empty_answers_are_skipped():
    row = _hc3_row()
    row["human_answers"] = ["   ", "real answer here"]
    assert len(parse_hc3([row])) == 2


# --------------------------------------------------------------- contamination

TRAIN_TEXT = (
    "The harbour authority published its annual dredging schedule in March, listing "
    "eleven separate operations across the estuary and the two tidal basins. Silt "
    "accumulation had increased noticeably since the previous survey."
)


def test_identical_text_is_caught_as_exact_contamination():
    rep = check("raid", [("t1", TRAIN_TEXT)], [("e1", TRAIN_TEXT)])
    assert not rep.clean and rep.exact_overlap == ["e1"]


def test_normalisation_matches_ingestion_so_markup_differences_still_collide():
    """A quick local `text.lower().strip()` here would pass while contamination is
    present, which is worse than no check at all."""
    wrapped = f"<div><p>{TRAIN_TEXT}</p></div>"
    assert canonical_hash(wrapped) == canonical_hash(TRAIN_TEXT)
    rep = check("raid", [("t1", TRAIN_TEXT)], [("e1", wrapped)])
    assert rep.exact_overlap == ["e1"]


def test_a_lightly_rewritten_document_is_caught_as_near_contamination():
    """FineWeb and RAID both draw on public web text. The same article with different
    boilerplate contaminates just as badly, and exact hashing misses every one.

    This pair sits at a true Jaccard of about 0.75, which dedup's 0.8 threshold misses.
    Contamination uses 0.5 precisely so it does not.
    """
    rewritten = TRAIN_TEXT + " A short additional sentence was appended by the site."
    rep = check("raid", [("t1", TRAIN_TEXT)], [("e1", rewritten)])
    assert not rep.clean
    assert rep.exact_overlap == [] and rep.near_overlap
    assert rep.near_overlap[0][1] == "t1"


def test_the_contamination_threshold_is_looser_than_dedups_on_purpose():
    """The error costs run in opposite directions. A dedup false positive destroys
    training data; a contamination false negative destroys the headline claim."""
    from forge.evaluation.contamination import (
        CONTAMINATION_NEAR_THRESHOLD,
        DEDUP_NEAR_THRESHOLD,
    )

    assert CONTAMINATION_NEAR_THRESHOLD < DEDUP_NEAR_THRESHOLD

    rewritten = TRAIN_TEXT + " A short additional sentence was appended by the site."
    strict = check("raid", [("t1", TRAIN_TEXT)], [("e1", rewritten)], near_threshold=0.8)
    loose = check("raid", [("t1", TRAIN_TEXT)], [("e1", rewritten)], near_threshold=0.5)
    assert strict.clean, "dedup's threshold would miss this overlap"
    assert not loose.clean, "the contamination threshold must catch it"


def test_unrelated_documents_are_clean():
    other = (
        "Financial regulators published a revised framework for capital adequacy "
        "requirements affecting mid-sized institutions across several jurisdictions. "
        "The directive takes effect at the start of the next reporting period."
    )
    assert check("mage", [("t1", TRAIN_TEXT)], [("e1", other)]).clean


def test_contamination_raises_rather_than_being_a_footnote():
    rep = check("raid", [("t1", TRAIN_TEXT)], [("e1", TRAIN_TEXT)])
    with pytest.raises(RuntimeError, match="contaminated"):
        rep.raise_if_contaminated()


def test_mage_machine_label_is_zero_as_measured() -> None:
    """THE BUG THIS FILE'S MODULE DOCSTRING PREDICTED, AND GOT BACKWARDS.

    parse_mage defaulted to machine_label=1. MAGE uses 0 for machine and 1 for human,
    measured over 6,000 rows by tabulating the label against the `src` field: every label-0
    source names a generator (eli5_machine_continuation_flan_t5_base,
    eli5_machine_specified_text-davinci-003) and every label-1 source names a human corpus
    (hswag_human, xsum_human, yelp_human). 100% against 0%.

    Nothing before the first scored run could have caught this. The parser was
    self-consistent, the schema check passed, the counts were healthy. An inverted label
    only shows up when scores meet labels, which is why assert_mage_polarity exists and why
    it stopped the evaluation rather than reporting a number.
    """
    from forge.evaluation.benchmarks import MAGE_MACHINE_LABEL, parse_mage

    assert MAGE_MACHINE_LABEL == 0

    rows = [
        {"text": "generated text", "label": 0, "src": "eli5_machine_continuation_flan_t5_base"},
        {"text": "a person wrote this", "label": 1, "src": "xsum_human"},
    ]
    docs = parse_mage(rows)
    by_src = {d.meta["src"]: d.label for d in docs}
    assert by_src["eli5_machine_continuation_flan_t5_base"] == 1, "generator row must be AI"
    assert by_src["xsum_human"] == 0, "human-corpus row must be human"


def test_the_polarity_guard_accepts_a_detector_that_works() -> None:
    """The accepting side. A guard that only ever fires is not a guard.

    Without this, flipping the default to silence the failure would look like a fix even if
    assert_mage_polarity had been broken into always raising.
    """
    from forge.evaluation.benchmarks import assert_mage_polarity, parse_mage

    rows = [
        {"text": "gen", "label": 0, "src": "eli5_machine_continuation_t0_11b"},
        {"text": "gen2", "label": 0, "src": "yelp_machine_specified_gpt-3.5-trubo"},
        {"text": "human", "label": 1, "src": "xsum_human"},
        {"text": "human2", "label": 1, "src": "wp_human"},
    ]
    docs = parse_mage(rows)
    scores = [0.93 if d.label == 1 else 0.04 for d in docs]
    assert_mage_polarity(docs, scores)          # must not raise
