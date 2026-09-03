"""The text arms must differ in exactly one place, and the budget must not be one of them.

WHY THIS FILE EXISTS. The arms are two YAML files that are meant to be identical except for
their AI source. Nothing enforces that. They have already drifted once: all three arm configs
pointed `paths.ai` at the same directory while `data.include` was decorative, which would have
trained Arm A and Arm B on identical data and supported the conclusion "matched mirrors make no
difference". Two successful runs, two plausible table rows, one false finding, no error.

`ai_cap` is the same hazard with a smaller blast radius. It is filled in by hand, from a count
taken after generation, into two files. Typing it into one and not the other, or typing two
different numbers, gives the larger arm more data. The measured gap then includes a volume
effect and the experiment stops answering its own question. A human reading two files side by
side is exactly the check that misses this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.common.config import REPO_ROOT, load

ARMS = (
    "configs/training/baseline_minimal.yaml",
    "configs/training/mirror_minimal.yaml",
    "configs/training/hard_negative_minimal.yaml",
)


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    return {path: load(path) for path in ARMS}


def test_both_arms_declare_a_budget(configs: dict[str, dict]) -> None:
    """A null cap is not a neutral default; it lets the larger arm train on more data."""
    for path, cfg in configs.items():
        assert cfg["data"].get("ai_cap") is not None, (
            f"{path} has no ai_cap. The arms would train on 18,856 and 22,713 documents, "
            "and any gap between them would be partly a volume effect."
        )


def test_the_arms_agree_on_the_budget(configs: dict[str, dict]) -> None:
    """The regression this file is for."""
    caps = {path: cfg["data"]["ai_cap"] for path, cfg in configs.items()}
    assert len(set(caps.values())) == 1, f"the arms carry different budgets: {caps}"


def test_the_budget_does_not_exceed_what_was_generated(configs: dict[str, dict]) -> None:
    """A cap above the smaller arm's yield silently does nothing.

    `load_examples` only caps when the arm has MORE rows than the cap. Set it above the
    smaller arm's yield and that arm passes through untouched while the larger one is cut,
    which is the unequal-budget bug wearing the clothes of a fix.

    MEASURED, not hardcoded. An earlier version of this test carried the literal 18,856,
    which was correct for exactly one corpus and became wrong the moment the corpora were
    regenerated: it then failed on a config that was fine. A test that has to be edited
    whenever the data changes is a test that will be edited to pass. This reads the parquet
    when it is there and skips when it is not, so it is silent on a laptop and binding on
    the machine that actually trains.
    """
    pytest.importorskip("pyarrow")
    import pyarrow.dataset as ds

    counts = {}
    for path, cfg in configs.items():
        root = REPO_ROOT / cfg["paths"]["ai"]
        if not root.is_dir():
            pytest.skip(f"{root} is not present; nothing to measure against")
        counts[path] = ds.dataset(str(root), format="parquet", partitioning="hive").count_rows()

    smallest = min(counts.values())
    for path, cfg in configs.items():
        assert cfg["data"]["ai_cap"] <= smallest, (
            f"{path} caps at {cfg['data']['ai_cap']}, above the smaller arm's {smallest} "
            f"generated documents ({counts}), so the cap does nothing for that arm"
        )


def test_the_arms_point_at_different_ai_sources(configs: dict[str, dict]) -> None:
    """The one place they are allowed to differ, and the place they once did not."""
    sources = {cfg["paths"]["ai"] for cfg in configs.values()}
    assert len(sources) == len(ARMS), f"both arms read the same AI directory: {sources}"


def test_each_arm_declares_which_arm_it_is(configs: dict[str, dict]) -> None:
    """Verified at load time against the prompt_version recorded on every document.

    The names are the GENERATION arm names, "random" and "mirror", not the training-config
    filenames. They have to be, because that is what `expect_arm` checks against the
    prompt_version stored on each document; a config saying "baseline" would fail at load.
    """
    arms = {cfg["data"]["arm"] for cfg in configs.values()}
    assert arms == {"random", "mirror", "hard_negative"}, arms


def test_both_arms_length_match_against_the_same_reference(configs: dict[str, dict]) -> None:
    """Pointing them at different references would reshape the arms differently.

    The mirror arm matching itself is a deliberate no-op; the control arm is reshaped to
    track it. Two different references would be two different reshapings and no comparison.
    """
    references = {cfg["data"].get("ai_reference") for cfg in configs.values()}
    assert len(references) == 1 and None not in references, (
        f"the arms length-match against different references: {references}"
    )


def test_the_arms_are_otherwise_identical(configs: dict[str, dict]) -> None:
    """Everything not deliberately different must actually be the same.

    Stated as a whitelist rather than a spot check: a future edit to one arm's learning rate
    or epoch count fails here rather than quietly becoming the explanation for a result.
    """
    allowed_to_differ = {"experiment", "paths", "data", "mining"}
    reference_path, *others = ARMS
    reference = configs[reference_path]
    for path in others:
        other = configs[path]
        for key in set(reference) | set(other):
            if key in allowed_to_differ:
                continue
            assert reference.get(key) == other.get(key), (
                f"{path} differs from {reference_path} in {key!r}, which is not one of the "
                f"treatment variables: {other.get(key)} vs {reference.get(key)}"
            )


def test_the_data_blocks_differ_only_where_they_must(configs: dict[str, dict]) -> None:
    reference_path, *others = ARMS
    reference = configs[reference_path]["data"]
    for path in others:
        other = configs[path]["data"]
        for key in set(reference) | set(other):
            if key == "arm":
                continue
            assert reference.get(key) == other.get(key), (
                f"data.{key} differs between {reference_path} and {path}: "
                f"{reference.get(key)} vs {other.get(key)}"
            )


def test_every_arm_config_named_here_exists() -> None:
    for path in ARMS:
        assert (REPO_ROOT / Path(path)).is_file(), f"{path} is missing"


def test_both_arms_cap_the_human_pool_identically(configs: dict[str, dict]) -> None:
    """human_cap controls RUN TIME, not the experiment, so it must be identical.

    Both arms read the same human corpus. Capping it differently would give one arm more
    negatives than the other, and a false-positive-rate comparison between arms trained on
    different amounts of human text says nothing.
    """
    caps = {path: cfg["data"].get("human_cap") for path, cfg in configs.items()}
    assert len(set(caps.values())) == 1, f"the arms cap humans differently: {caps}"


def test_both_arms_run_the_same_number_of_epochs(configs: dict[str, dict]) -> None:
    """Already covered by the identical-training-block assertion, stated separately because
    epochs is the field most likely to be edited by hand at 2am."""
    epochs = {cfg["training"]["epochs"] for cfg in configs.values()}
    assert len(epochs) == 1, f"the arms train for different numbers of epochs: {epochs}"


def test_arm_c_does_not_read_another_arms_ai_data(configs: dict[str, dict]) -> None:
    """The regression. Arm C pointed at data/silver/mirrors, which is arm B's directory.

    It sat directly under the comment saying each arm must differ. Left there, arm C would
    have retrained arm B on arm B's data and the difference between the two runs would have
    been reported as a hard-negative effect. Nothing would have errored.
    """
    c = configs["configs/training/hard_negative_minimal.yaml"]["paths"]["ai"]
    others = {
        configs[p]["paths"]["ai"]
        for p in ARMS
        if p != "configs/training/hard_negative_minimal.yaml"
    }
    assert c not in others, f"arm C reads {c}, which belongs to another arm"


def test_arm_c_carries_the_shared_budget_not_its_own(configs: dict[str, dict]) -> None:
    """Mined humans and targeted mirrors REPLACE part of a fixed budget, never extend it.

    Mining draws from the reserve pool, which sits outside train/val/test, so mined
    documents are additional unless the caps hold. An arm C trained on more human text than
    arms A and B would show a better false-positive rate for a reason that has nothing to do
    with failure-driven selection.
    """
    c = configs["configs/training/hard_negative_minimal.yaml"]["data"]
    for path in ARMS:
        other = configs[path]["data"]
        assert c["ai_cap"] == other["ai_cap"], f"ai_cap differs from {path}"
        assert c["human_cap"] == other["human_cap"], f"human_cap differs from {path}"


def test_arm_c_mines_a_named_arm_that_exists(configs: dict[str, dict]) -> None:
    """`from_arm` decides whose failures define the treatment. It cannot be implicit."""
    mining = configs["configs/training/hard_negative_minimal.yaml"].get("mining", {})
    assert mining.get("from_arm"), "arm C does not say whose failures it mines"
    declared = {configs[p]["data"]["arm"] for p in ARMS}
    assert mining["from_arm"] in declared, (
        f"arm C mines {mining['from_arm']!r}, which is not one of {sorted(declared)}"
    )
