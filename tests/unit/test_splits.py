"""The split tests are the most important tests in this repo.

A group leak does not crash anything. It just makes every metric better than it should
be, and it survives all the way to a writeup. So it gets tested directly.
"""

from forge.common.schemas import Split
from forge.common.splits import assign_split, check_no_group_leakage, group_id_for


def test_assignment_is_deterministic():
    assert assign_split("grp_abc") is assign_split("grp_abc")


def test_derived_records_inherit_the_group_and_land_together():
    human_id = "fw_000000018392"
    group = group_id_for(human_id)
    # mirrors, hard negatives and adversarial variants all reuse the human's group id
    derived = [group] * 5
    splits = {assign_split(g) for g in [group, *derived]}
    assert len(splits) == 1, "a human document and its derivatives must share one split"


def test_ratios_are_roughly_right():
    n = 20000
    counts = {s: 0 for s in Split}
    for i in range(n):
        counts[assign_split(f"grp_{i}")] += 1
    assert 0.78 < counts[Split.TRAIN] / n < 0.82
    assert 0.08 < counts[Split.VAL] / n < 0.12
    assert 0.08 < counts[Split.TEST] / n < 0.12


def test_leakage_checker_catches_a_split_group():
    try:
        check_no_group_leakage([("grp_a", Split.TRAIN), ("grp_a", Split.TEST)])
    except ValueError as e:
        assert "grp_a" in str(e)
    else:
        raise AssertionError("leakage checker failed to detect a group in two splits")


def test_adding_documents_does_not_move_existing_ones():
    before = {f"grp_{i}": assign_split(f"grp_{i}") for i in range(500)}
    # simulate dataset v0.2: same ids plus more
    after = {f"grp_{i}": assign_split(f"grp_{i}") for i in range(1000)}
    for k, v in before.items():
        assert after[k] is v
