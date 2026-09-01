"""The Makefile must run on a machine that has no .venv.

WHY THIS EXISTS. Every recipe used to begin with `./.venv/bin/python`, which is
the layout `make setup` produces on a laptop. On the rented GPU pod the package
is installed into the image's own site-packages and there is no .venv, so
`make min-ingest` died with

    make: ./.venv/bin/python: No such file or directory

before executing a single line of forge code. The full test suite passed on that
same pod at that same moment, because the suite was invoked as `python -m pytest`
directly and no test had ever exercised a make target. That is the actual mistake:
the build entry points were the one part of the repo with no coverage at all, so
an assumption about directory layout could sit in them undetected across both
machines.

These tests are cheap text assertions rather than subprocess runs, because
running the real targets would download corpora.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"


def _recipe_lines() -> list[tuple[int, str]]:
    """Return (line number, text) for every recipe line, which is TAB-indented."""
    lines = MAKEFILE.read_text().split("\n")
    return [(i + 1, ln) for i, ln in enumerate(lines) if ln.startswith("\t")]


def test_makefile_exists() -> None:
    assert MAKEFILE.is_file(), f"expected a Makefile at {MAKEFILE}"


def test_no_recipe_hardcodes_the_venv_interpreter() -> None:
    """The bug, stated directly.

    `setup` is the one target allowed to name the venv, because creating it is
    what that target is for. Everywhere else the interpreter must come from a
    variable so a machine without a .venv can still run the target.
    """
    offenders = [
        (n, ln)
        for n, ln in _recipe_lines()
        if re.search(r"\.venv/bin/", ln) and "$(VENV)" not in ln
    ]
    assert not offenders, (
        "these recipe lines hardcode the venv interpreter and will fail on a "
        f"machine that has no .venv: {offenders}"
    )


def test_interpreter_variable_falls_back_to_path_python() -> None:
    """A variable alone is not enough; it has to degrade to a plain `python`.

    Defining PY := .venv/bin/python would satisfy the test above while still
    being broken on the pod, so assert the conditional fallback specifically.
    """
    text = MAKEFILE.read_text()
    assert re.search(r"^PY\s*:?=", text, re.MULTILINE), "no PY variable defined"
    py_line = next(ln for ln in text.split("\n") if re.match(r"^PY\s*:?=", ln))
    assert "wildcard" in py_line and py_line.rstrip().endswith("python)"), (
        "PY must select the venv only when it exists and otherwise fall back to "
        f"PATH python; got: {py_line!r}"
    )


@pytest.mark.parametrize(
    "target",
    ["min-ingest", "min-mirror", "min-smoke", "min-train", "test", "spec"],
)
def test_tier_one_targets_are_defined(target: str) -> None:
    """The targets the first result depends on must still exist by these names.

    The runbook and docs/FINISH.md instruct a human to type these. A rename that
    silently orphans one of them would surface as a confusing failure on a paid
    GPU rather than here.
    """
    text = MAKEFILE.read_text()
    assert re.search(rf"^{re.escape(target)}:", text, re.MULTILINE), (
        f"target {target!r} is referenced by the runbook but not defined"
    )
