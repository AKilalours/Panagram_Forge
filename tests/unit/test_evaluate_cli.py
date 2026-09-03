"""`forge evaluate` must run the adversarial lab.

It was `_not_yet("3", "the evaluation lab")` while attacks.py and lab.py were both written
and tested, so the laboratory existed and nothing could reach it. The first test fails
against that version.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from forge.cli import app

runner = CliRunner()
CONFIG = "configs/training/mirror_minimal.yaml"

AI_TEXT = (
    "The committee reviewed the quarterly figures and concluded that the observed "
    "variance in the reported totals arises from a change in the accounting treatment "
    "of deferred revenue rather than from any underlying shift in customer demand. "
)


class FakeArm:
    """A detector that is confident until the text is perturbed at all."""

    experiment = "forge_min_mirror"

    class policy:
        model_version = "forge-test@deadbeef"
        threshold = 0.5
        fpr_budget = 0.001

    def __init__(self) -> None:
        self.seen = 0

    def score(self, text: str):
        self.seen += 1
        clean = text.strip() == AI_TEXT.strip() * 3
        return type("S", (), {"mean": 0.95 if clean else 0.30})()


class FakeExample:
    def __init__(self, i: int) -> None:
        self.doc_id, self.label, self.text = f"ai_{i}", 1, AI_TEXT * 3


@pytest.fixture
def wired(monkeypatch):
    import forge.cli as C

    arm = FakeArm()
    monkeypatch.setattr("forge.inference.scorer.load_arm", lambda *_a, **_k: arm)
    monkeypatch.setattr(
        "forge.training.data.load_examples",
        lambda **_k: [FakeExample(i) for i in range(12)],
    )
    return C, arm


def test_evaluate_runs_the_lab_and_writes_the_table(wired, tmp_path):
    """THE REGRESSION TEST. The old command raised PhaseNotImplemented every time."""
    result = runner.invoke(
        app, ["evaluate", "--config", CONFIG, "--out", str(tmp_path), "--limit", "12"]
    )
    assert result.exit_code == 0, result.output

    written = tmp_path / "adversarial_forge_min_mirror.json"
    assert written.exists(), "the lab produced no table"
    payload = json.loads(written.read_text())
    assert payload["n_ai_documents"] == 12
    assert payload["split"] == "test"
    assert payload["results"], "no attacks were run"


def test_the_table_reports_both_conditions_for_every_attack(wired, tmp_path):
    """Raw and preprocessed. The gap between them IS the value of normalisation."""
    runner.invoke(app, ["evaluate", "--config", CONFIG, "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "adversarial_forge_min_mirror.json").read_text())
    for r in payload["results"]:
        assert "delta_fnr_raw" in r and "delta_fnr_preprocessed" in r
        assert "n_noop" in r and "n_invalid" in r, (
            "no-ops and meaning-breaking attacks must be counted, not silently scored"
        )


def test_the_run_records_the_threshold_it_attacked(wired, tmp_path):
    """A delta-FNR without its operating point is uninterpretable."""
    runner.invoke(app, ["evaluate", "--config", CONFIG, "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "adversarial_forge_min_mirror.json").read_text())
    assert payload["threshold"] == 0.5
    assert payload["fpr_budget"] == 0.001
    assert payload["model_version"]


def test_no_detector_means_no_table(monkeypatch, tmp_path):
    from forge.inference.scorer import ArmUnavailable

    def _no_weights(*_a, **_k):
        raise ArmUnavailable("no checkpoint at outputs/forge_min_mirror/best.pt")

    monkeypatch.setattr("forge.inference.scorer.load_arm", _no_weights)
    result = runner.invoke(app, ["evaluate", "--config", CONFIG, "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert not list(tmp_path.glob("*.json"))


def test_no_ai_documents_means_no_table(monkeypatch, tmp_path):
    """FNR is the metric an evader moves, and FNR needs AI documents."""
    monkeypatch.setattr("forge.inference.scorer.load_arm", lambda *_a, **_k: FakeArm())
    monkeypatch.setattr("forge.training.data.load_examples", lambda **_k: [])
    result = runner.invoke(app, ["evaluate", "--config", CONFIG, "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert not list(tmp_path.glob("*.json"))
