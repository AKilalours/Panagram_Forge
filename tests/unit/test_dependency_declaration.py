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


# --------------------------------------------------------------- the serve extra must serve

# The modules a running server reaches: the HTTP layer, and the two detectors plus the
# attribution pass it calls into. Add a module here when the server starts importing it.
SERVING_MODULES = (
    "api/forge_app.py",
    "api/results.py",
    "src/forge/inference/scorer.py",
    "src/forge/inference/decision.py",
    "src/forge/image/detector.py",
    "src/forge/image/attribution.py",
    "src/forge/image/report.py",
    "src/forge/image/evidence.py",
    "src/forge/image/maps.py",
)

# Import name to distribution name, for the few where they differ.
DISTRIBUTION = {"PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn"}


def _third_party_imports(path: Path) -> set[str]:
    """Every non-stdlib top-level import, including imports inside functions.

    Function-level imports matter more than usual here: the server defers torch and
    transformers into request handlers to keep startup fast, so a naive scan of the file
    header would have found nothing and concluded the serve extra was complete.
    """
    import ast
    import sys

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return {
        name for name in found
        if name not in sys.stdlib_module_names and name not in {"forge", "api", "__future__"}
    }


def test_the_serve_extra_can_actually_serve():
    """THE REGRESSION. `pip install -e '.[serve,image,dev]'` produced a server with no models.

    torch and transformers were declared only under `train`, because they were thought of
    as training dependencies. They are not: the text arms, the visual detector and the
    occlusion attribution pass all run on them at request time. The app booted normally and
    every detector reported "not loaded", so the interface looked like a working build with
    a configuration problem, which is the most expensive way for this to fail.

    Deriving the list from the code rather than restating it means the next module the
    server starts importing is checked automatically.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    serve = {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"]["optional-dependencies"]["serve"]
    }
    serve |= {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"].get("dependencies", [])
    }

    needed: dict[str, str] = {}
    for relative in SERVING_MODULES:
        path = ROOT / relative
        if not path.exists():
            continue
        for name in _third_party_imports(path):
            needed.setdefault(_normalise(DISTRIBUTION.get(name, name)), relative)

    missing = sorted(f"{dist} (imported by {where})" for dist, where in needed.items()
                     if dist not in serve)
    assert not missing, (
        "the serve extra does not declare what the serving path imports: "
        + "; ".join(missing)
        + ". Installing '.[serve]' would give a server that boots and cannot load a model."
    )


def test_the_tokenizer_backend_is_declared_wherever_the_text_arm_is_served():
    """THE REGRESSION. torch and transformers were present and the text arm still failed.

    The DeBERTa-v3 backbone tokenizer is a SentencePiece model. Without `sentencepiece`,
    transformers falls back to a TikToken converter, that import fails too, and
    AutoTokenizer.from_pretrained raises. The arm reports unavailable on a machine that has
    the entire model stack installed, which reads as a broken checkpoint rather than a
    missing package.

    It is not discoverable by scanning imports: no file in this project imports
    sentencepiece. It is a runtime backend transformers reaches for, so it has to be
    asserted by name in both places the app is installed from.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    serve = {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"]["optional-dependencies"]["serve"]
    }
    assert "sentencepiece" in serve, "the text arm cannot build its tokenizer without it"

    space = _requirement_names((ROOT / "space" / "requirements.txt").read_text(encoding="utf-8"))
    assert "sentencepiece" in space, "the deployed Space would serve a dead text tab"


def test_transformers_is_pinned_below_5_in_both_installs():
    """The arms' thresholds were fitted under transformers 4.x tokenisation.

    Version 5 rewrote tokenizer construction. A threshold is only meaningful against the
    tokenisation it was measured with, so an unpinned upper bound silently swaps the
    tokeniser under a fixed decision boundary. Unpin this only together with a re-run of
    the evaluation and a new record under reports/experiments.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = [
        spec for group in
        [data["project"].get("dependencies", [])]
        + list(data["project"].get("optional-dependencies", {}).values())
        for spec in group
        if _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]) == "transformers"
    ]
    assert specs, "transformers is not declared anywhere"
    for spec in specs:
        assert "<5" in spec, f"unbounded transformers requirement: {spec!r}"

    space_text = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    line = [ln for ln in space_text.splitlines() if ln.strip().startswith("transformers")]
    assert line and "<5" in line[0], f"the Space would install transformers 5: {line}"


def test_huggingface_hub_is_pinned_below_1_in_both_installs():
    """THE REGRESSION. `pip install -U huggingface_hub` installed 1.30 and broke the venv.

    transformers 4.x requires huggingface-hub<1.0. Both requirements were written with a
    lower bound and no ceiling, so the resolver was free to install a major version the
    pinned transformers cannot use. pip printed the conflict and installed it anyway.

    Locally that is a visible error. In the Space build it would be a container that starts,
    reports the text arms unavailable, and gives no obvious reason, on a machine nobody is
    watching.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = [
        spec for group in
        [data["project"].get("dependencies", [])]
        + list(data["project"].get("optional-dependencies", {}).values())
        for spec in group
        if _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]) == "huggingface-hub"
    ]
    assert specs, "huggingface-hub is not declared anywhere"
    for spec in specs:
        assert "<1" in spec, f"unbounded huggingface-hub requirement: {spec!r}"

    space_text = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    line = [ln for ln in space_text.splitlines() if ln.strip().startswith("huggingface_hub")]
    assert line and "<1.0" in line[0], f"the Space would install huggingface_hub 1.x: {line}"


# ------------------------------------------------------- the two deployment requirement files

def test_the_streamlit_requirements_agree_with_the_space_requirements():
    """Two deployment files, one set of pins. They must not drift.

    Streamlit Community Cloud reads requirements.txt at the repository root and nothing
    else, so the pins that keep the text arm alive (transformers below 5, sentencepiece,
    huggingface_hub below 1) have to be repeated there. A repeated pin is a pin that rots:
    this project has already shipped python-multipart in one file and not the other, and a
    dead text tab is exactly what that produces.

    Shared packages must carry identical specifiers. Either file may add packages the other
    does not need, which is how streamlit and onnxruntime differ legitimately.
    """
    def specs(text: str) -> dict[str, str]:
        out = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0]
            out[_normalise(name)] = line[len(name):].strip()
        return out

    root = specs((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    space = specs((ROOT / "space" / "requirements.txt").read_text(encoding="utf-8"))

    disagreements = {
        name: (root[name], space[name])
        for name in set(root) & set(space)
        if root[name] != space[name]
    }
    assert not disagreements, (
        "requirements.txt and space/requirements.txt disagree on shared packages: "
        f"{disagreements}"
    )
    for required in ("torch", "transformers", "sentencepiece", "huggingface-hub", "pillow"):
        assert required in root, f"{required} missing from the Streamlit requirements"
    assert "streamlit" in root, "the Streamlit deployment cannot run without streamlit"


def test_the_streamlit_app_declares_the_same_ceilings():
    """The pins that matter are the upper bounds, so assert them by name in the root file."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = {ln.split("#")[0].strip() for ln in text.splitlines()}
    assert any(ln.startswith("transformers") and "<5" in ln for ln in lines)
    assert any(ln.startswith("huggingface_hub") and "<1.0" in ln for ln in lines)
