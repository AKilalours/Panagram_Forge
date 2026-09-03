"""THE REGRESSION. Two hand-maintained dependency lists disagreed, and the wrong one won.

`space/requirements.txt` listed `python-multipart`. The `serve` extra in `pyproject.toml`
did not. FastAPI needs it to build the route for `/v1/image/analyze`, and it raises at
IMPORT time rather than at request time, so the failure is not a degraded endpoint: the
process does not start. That meant `pip install -e ".[serve]"` produced an installation
that could not boot the app, while the deployment target booted fine. The package was
wrong and the thing furthest from the tests was right.

The fix is not "add the package". It is to make the two lists unable to disagree: anything
the deployed Space installs must also be declared in the package, so a dependency added for
deployment cannot skip the package that developers and CI install from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:  # 3.11+ ships tomllib; the package supports 3.10, where tomli is the equivalent.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter, not the code
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli on 3.10")

ROOT = Path(__file__).resolve().parents[2]


def _normalise(name: str) -> str:
    """PEP 503 names: case-insensitive, and -, _ and . are all the same character."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(text: str) -> set[str]:
    names = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0]
        if name:
            names.add(_normalise(name))
    return names


def _declared_names() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    lists = [project.get("dependencies", [])]
    lists += list(project.get("optional-dependencies", {}).values())
    names = set()
    for group in lists:
        for spec in group:
            names.add(_normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]))
    return names


def test_everything_the_space_installs_is_declared_in_the_package():
    space = _requirement_names((ROOT / "space" / "requirements.txt").read_text(encoding="utf-8"))
    missing = sorted(space - _declared_names())
    assert not missing, (
        "space/requirements.txt installs packages pyproject.toml never declares: "
        f"{missing}. A developer running `pip install -e '.[serve,image,train]'` gets an "
        "installation the deployed app does not match."
    )


def test_multipart_is_a_declared_serve_dependency():
    """Named explicitly, because its absence is a boot failure and not a missing feature."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    serve = {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"]["optional-dependencies"]["serve"]
    }
    assert "python-multipart" in serve, (
        "fastapi raises at import time on a multipart route without it; the app will not start"
    )
    assert "pillow" in serve, "every image upload is decoded with Pillow on the serve path"
