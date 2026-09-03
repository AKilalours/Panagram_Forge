"""The results page reads records; it never invents or hardcodes a number.

A results table maintained by hand drifts away from the runs it describes, and the drift is
invisible: every figure still looks like a figure. So this endpoint reads the committed run
records and nothing else, and a missing record produces a missing section rather than a zero.
"""

from __future__ import annotations

import json

import pytest


def test_absent_records_produce_absent_sections(tmp_path, monkeypatch):
    """A zero in a results table is a claim. An absent section is not."""
    monkeypatch.chdir(tmp_path)
    import importlib

    import api.results as R

    importlib.reload(R)
    built = R.build()
    assert built["in_distribution"] is None
    assert built["out_of_distribution"] is None
    assert built["image_probe"] is None


def test_an_unreadable_record_is_absent_rather_than_partly_believed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib

    import api.results as R

    importlib.reload(R)
    R.ROOT.mkdir(parents=True)
    (R.ROOT / "image_detector_polarity.json").write_text("{ truncated")
    assert R.build()["image_probe"] is None


def test_every_figure_comes_from_the_record(tmp_path, monkeypatch):
    """Values are passed through, not recomputed and not rounded into something else."""
    monkeypatch.chdir(tmp_path)
    import importlib

    import api.results as R

    importlib.reload(R)
    R.ROOT.mkdir(parents=True)
    (R.ROOT / "forge_min_mirror.json").write_text(json.dumps({
        "val": {"fpr_at_budget": 0.000502, "fnr": 0.004304, "auroc": 0.999971,
                "ece": 0.003713, "n_human": 1993, "n_ai": 2091, "threshold": 0.992285}
    }))
    row = R.build()["in_distribution"]["rows"][0]
    assert row["fnr"] == 0.004304
    assert row["auroc"] == 0.999971
    assert row["n_ai"] == 2091


def test_the_real_records_load(tmp_path):
    """Runs against the repository's own records when they are present."""
    import pathlib

    import api.results as R

    if not (R.ROOT / "ood_summary.json").exists():
        pytest.skip("no committed OOD record here")
    built = R.build()
    assert len(built["out_of_distribution"]["cells"]) == 6
    assert {c["benchmark"] for c in built["out_of_distribution"]["cells"]} == {
        "hc3", "mage", "raid"
    }
    assert pathlib.Path("reports/experiments").exists()


def test_the_image_probe_carries_its_in_sample_flag():
    """The operating point is fitted in sample. The page must be able to say so."""
    import api.results as R

    if not (R.ROOT / "image_detector_polarity.json").exists():
        pytest.skip("no probe record here")
    probe = R.build()["image_probe"]
    assert probe["in_sample"] is True
    assert probe["recall"] is not None
    assert probe["threshold"] is not None
