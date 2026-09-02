"""The two text arms must differ in exactly one place, and the budget must not be one of them.

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

ARMS = ("configs/training/baseline_minimal.yaml", "configs/training/mirror_minimal.yaml")


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
    assert arms == {"random", "mirror"}, arms


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
    allowed_to_differ = {"experiment", "paths", "data"}
    baseline, mirror = (configs[path] for path in ARMS)
    for key in set(baseline) | set(mirror):
        if key in allowed_to_differ:
            continue
        assert baseline.get(key) == mirror.get(key), (
            f"the arms differ in {key!r}, which is not one of the treatment variables: "
            f"{baseline.get(key)} vs {mirror.get(key)}"
        )


def test_the_data_blocks_differ_only_where_they_must(configs: dict[str, dict]) -> None:
    baseline, mirror = (configs[path]["data"] for path in ARMS)
    for key in set(baseline) | set(mirror):
        if key == "arm":
            continue
        assert baseline.get(key) == mirror.get(key), (
            f"data.{key} differs between arms: {baseline.get(key)} vs {mirror.get(key)}"
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
