"""Arm A, the control arm. These tests exist to stop the comparison being rigged."""

import pytest

from forge.common.config import load
from forge.generation.assignment import held_out_families, parse_roster
from forge.generation.random_synthetic import (
    TOPICS,
    generate_random,
    length_pool_from_corpus,
    spec_for,
)

POOL = [150, 200, 250, 300, 350, 400]


def _cfg():
    return load("configs/generation/generators_minimal.yaml")


def test_random_documents_are_not_matched_to_anything():
    """The whole point of the control. topic_match, length_match and style_match must all
    be False, or Arm A is a weak mirror rather than a random baseline."""
    res = generate_random(20, POOL, _cfg(), backend="fake")
    assert res.docs
    for d in res.docs:
        assert d.mirror.topic_match is False
        assert d.mirror.length_match is False
        assert d.mirror.style_match is False
        assert d.mirror.attributes["arm"] == "random"


def test_topics_come_from_the_fixed_inventory_not_the_corpus():
    """Sampling topics from the human corpus would make this a mirror, not a control."""
    res = generate_random(30, POOL, _cfg(), backend="fake")
    assert all(d.mirror.attributes["topic"] in TOPICS for d in res.docs)


def test_length_distribution_matches_the_corpus_range_without_matching_documents():
    """Arm A must live in the same length RANGE, or the detector learns length and Arm A
    collapses for a reason unrelated to topic matching."""
    res = generate_random(40, POOL, _cfg(), backend="fake")
    targets = {d.mirror.target_tokens for d in res.docs}
    assert targets <= set(POOL)
    assert len(targets) > 1, "a single target length would be its own artifact"


def test_generation_is_deterministic_so_a_partial_run_resumes():
    assert spec_for(7, POOL) == spec_for(7, POOL)
    a = generate_random(10, POOL, _cfg(), backend="fake")
    b = generate_random(10, POOL, _cfg(), backend="fake")
    assert [d.text for d in a.docs] == [d.text for d in b.docs]


def test_only_held_in_families_generate_the_control_arm():
    """Same rule as mirrors: a held-out family in ANY training arm invalidates R3."""
    res = generate_random(40, POOL, _cfg(), backend="fake")
    held_out = {f.family for f in held_out_families(parse_roster(_cfg()))}
    assert not ({d.generator.family for d in res.docs} & held_out)


def test_the_control_arm_uses_the_same_generator_families_as_mirrors():
    """If mirrors win on generator diversity the result says nothing about matching."""
    from forge.generation.assignment import held_in_families

    res = generate_random(60, POOL, _cfg(), backend="fake")
    expected = {f.family for f in held_in_families(parse_roster(_cfg()))}
    assert set(res.stats["families_used"]) == expected


def test_random_documents_get_their_own_groups():
    """They have no human source, so they cannot inherit a group. Sharing one would put
    unrelated documents in the same split bucket."""
    res = generate_random(20, POOL, _cfg(), backend="fake")
    assert len({d.source_group_id for d in res.docs}) == len(res.docs)


def test_assistant_preamble_is_rejected_here_too():
    """Arm A must be held to the same quality bar as Arm B. A control polluted with chat
    formatting would lose for the wrong reason and flatter the mirror arm."""
    from forge.generation.generators.base import Decoding

    class Preambler:
        family, model_id, revision = "bad", "forge/bad", "v1"

        def generate(self, prompts, decoding: Decoding):
            return ["Certainly! Here is your text:\n\n" + " ".join(["word"] * 250)] * len(prompts)

    import forge.generation.run as run_mod

    orig = run_mod.build_generator
    run_mod.build_generator = lambda spec, backend: Preambler()
    try:
        res = generate_random(5, POOL, _cfg(), backend="fake")
        assert res.docs == []
        assert res.stats["rejected"]["assistant_preamble"] > 0
    finally:
        run_mod.build_generator = orig


def test_length_pool_is_read_from_the_real_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        length_pool_from_corpus(tmp_path)


def test_empty_length_pool_is_refused():
    with pytest.raises(ValueError):
        generate_random(5, [], _cfg(), backend="fake")
