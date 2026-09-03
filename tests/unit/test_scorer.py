"""The serving path must refuse rather than degrade.

These run without torch, transformers or weights. They cover the part that decides whether
a user sees a number at all, which is the part that has to be right even on a machine that
cannot load a model.
"""

from __future__ import annotations

import json
import sys

import pytest

from forge.inference.scorer import ARMS, ArmUnavailable, load_arm

@pytest.fixture(autouse=True)
def _clear_cache():
    load_arm.cache_clear()
    yield
    load_arm.cache_clear()


def test_an_unknown_arm_is_refused_by_name():
    with pytest.raises(ArmUnavailable) as e:
        load_arm("mirrors")           # plausible typo for "mirror"
    assert "unknown arm" in str(e.value)


def test_a_missing_checkpoint_names_the_path_it_wanted(tmp_path, monkeypatch):
    """The message has to be actionable. 'not available' sends a user nowhere."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ArmUnavailable) as e:
        load_arm("mirror")
    assert "no checkpoint" in str(e.value) or "configs/training" in str(e.value)


def test_a_missing_summary_is_refused_even_though_weights_exist(tmp_path, monkeypatch):
    """THE ONE THAT MATTERS. Weights without a threshold must not serve.

    A checkpoint alone is enough to produce a probability. Turning that probability into
    "human" or "AI" needs the threshold fit on validation at the 0.1% FPR budget. Falling
    back to 0.5 would look like it worked and would silently move the false-positive rate
    this whole project is built around.
    """
    import shutil
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for d in ("configs", "src"):
        shutil.copytree(repo / d, tmp_path / d, dirs_exist_ok=True)
    out = tmp_path / "outputs" / "forge_min_mirror"
    out.mkdir(parents=True)
    (out / "best.pt").write_bytes(b"not a real checkpoint, and never loaded")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ArmUnavailable) as e:
        load_arm("mirror")
    message = str(e.value)
    assert "summary.json" in message
    assert "threshold" in message


def test_available_reports_every_arm_and_never_raises(tmp_path, monkeypatch):
    """The UI calls this to render state. It must not be able to take the page down."""
    monkeypatch.chdir(tmp_path)
    from forge.inference.scorer import available

    state = available()
    assert set(state) == set(ARMS)
    assert all(isinstance(v, str) and v for v in state.values())


def test_the_arm_names_match_the_configs_that_exist():
    """Guards against the arm list drifting from configs/training/*_minimal.yaml."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for arm in ARMS:
        assert (repo / "configs" / "training" / f"{arm}_minimal.yaml").exists(), arm


def test_the_module_imports_without_torch(monkeypatch):
    """Serving code must not drag torch into every process that imports the package."""
    for name in list(sys.modules):
        if name.startswith("forge.inference.scorer"):
            del sys.modules[name]
    monkeypatch.setitem(sys.modules, "torch", None)
    import forge.inference.scorer as s

    assert s.ARMS == ("baseline", "mirror")
