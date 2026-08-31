"""spec-check is a CI gate, so it needs its own test."""

from forge.common.config import spec_check


def test_committed_configs_match_the_frozen_spec():
    problems = spec_check()
    assert problems == [], "configs drifted from data_spec_v1:\n" + "\n".join(problems)
