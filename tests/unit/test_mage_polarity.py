"""Guards the inverted-label landmine recorded in data_spec_v1 section 1.1."""

import pytest

from forge.evaluation.ood import assert_label_polarity


def test_correct_polarity_passes():
    assert_label_polarity(known_human_scores=[0.02, 0.05], known_machine_scores=[0.9, 0.95])


def test_inverted_polarity_fails_loudly():
    with pytest.raises(AssertionError):
        assert_label_polarity(known_human_scores=[0.9, 0.95], known_machine_scores=[0.02, 0.05])
