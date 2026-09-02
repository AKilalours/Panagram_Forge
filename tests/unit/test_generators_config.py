"""A mirror config's declared generator roster must be the one that gets used.

THE BUG THIS CAME FROM. `run()` hardcoded

    generators_cfg = load("configs/generation/generators.yaml")

while every mirror config declares its own `generators_config:` key. So running

    forge mirror --config configs/generation/mirror_minimal.yaml

loaded the FULL-scale roster: different model families, at unpinned revisions,
than the config said. Nothing errored at that point. The run was only stopped
much further downstream by require_pinned_revision, a guard written for an
unrelated purpose that happened to trip on the wrong roster's placeholder.

Had the full-scale roster been pinned, the probe would have generated mirrors
from the wrong models and reported success, and the dataset's own metadata would
have named a roster it never used.

This is the third instance of the same shape in this project: a config value that
is declared, parsed, and then never read. The other two were the cleaning policy
that never reached the Cleaner, and the two training arms whose `data.include`
was decorative while both read the same directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.common.config import REPO_ROOT, load
from forge.generation.run import DEFAULT_GENERATORS_CONFIG, resolve_generators_config


def test_falls_back_to_the_default_when_the_key_is_absent() -> None:
    assert resolve_generators_config({}) == DEFAULT_GENERATORS_CONFIG


def test_uses_the_roster_the_config_names() -> None:
    """The bug, stated directly."""
    cfg = {"generators_config": "configs/generation/generators_minimal.yaml"}
    assert resolve_generators_config(cfg) == "configs/generation/generators_minimal.yaml"
    assert resolve_generators_config(cfg) != DEFAULT_GENERATORS_CONFIG


def test_a_named_roster_that_does_not_exist_is_an_error_not_a_fallback() -> None:
    """Silently falling back would reintroduce the bug under a different cause.

    A typo in the path must not quietly resolve to the full-scale roster.
    """
    with pytest.raises(FileNotFoundError) as e:
        resolve_generators_config({"generators_config": "configs/generation/nope.yaml"})
    assert "nope.yaml" in str(e.value)


@pytest.mark.parametrize("bad", ["", "   ", 42, [], {}])
def test_rejects_a_non_path_value(bad: object) -> None:
    with pytest.raises(ValueError):
        resolve_generators_config({"generators_config": bad})


def test_the_minimal_mirror_config_resolves_to_the_minimal_roster() -> None:
    """End-to-end on the real files. This is the assertion that would have caught it.

    The unit tests above pass on a dict. This one reads the config that a human
    actually types on the command line and checks the roster it lands on, which is
    the level the bug lived at.
    """
    cfg = load("configs/generation/mirror_minimal.yaml")
    resolved = resolve_generators_config(cfg)
    assert resolved == "configs/generation/generators_minimal.yaml"
    assert (REPO_ROOT / resolved).is_file()


def test_every_mirror_config_names_a_roster_that_exists() -> None:
    """Guards against a future mirror config pointing at a deleted or renamed roster."""
    configs = sorted(Path(REPO_ROOT / "configs/generation").glob("mirror*.yaml"))
    assert configs, "expected at least one mirror config"
    for path in configs:
        rel = str(path.relative_to(REPO_ROOT))
        resolved = resolve_generators_config(load(rel))
        assert (REPO_ROOT / resolved).is_file(), f"{rel} names a missing roster {resolved}"


def test_the_minimal_roster_is_fully_pinned() -> None:
    """The roster the minimal arm uses must carry no placeholder revisions.

    require_pinned_revision enforces this at generation time, on a GPU, after a
    model download has already started. Asserting it here fails in milliseconds
    instead.
    """
    from forge.generation.pinning import unpinned_lines

    text = (REPO_ROOT / "configs/generation/generators_minimal.yaml").read_text()
    assert unpinned_lines(text) == []
