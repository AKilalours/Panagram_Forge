"""The results endpoint: measured evidence, read from committed run records.

Nothing here is computed at request time and nothing is hardcoded. Every number is read
from `reports/experiments/`, written by the run that produced it, so the page cannot drift
away from the repository the way a hand-maintained results table always does.

A file that is absent produces an absent section, never a zero and never a placeholder.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("reports/experiments")
BENCHMARKS = ("hc3", "mage", "raid")
ARMS = {"baseline": "A: random synthetic", "mirror": "B: matched mirrors"}


def _read(name: str) -> dict | None:
    path = ROOT / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - an unreadable record is an absent record
        return None


def _in_distribution() -> dict | None:
    rows = []
    for arm, label in ARMS.items():
        summary = _read(f"forge_min_{arm}.json")
        if not summary:
            continue
        val = summary.get("val", {})
        rows.append({
            "arm": arm, "label": label,
            "fpr": val.get("fpr_at_budget"), "fnr": val.get("fnr"),
            "auroc": val.get("auroc"), "ece": val.get("ece"),
            "n_human": val.get("n_human"), "n_ai": val.get("n_ai"),
            "threshold": val.get("threshold"),
        })
    if not rows:
        return None

    tier1 = _read("tier1_comparison.json") or {}
    return {
        "rows": rows,
        "significance": tier1.get("significance"),
        "caveat": tier1.get("caveat") or tier1.get("in_distribution_only"),
    }


def _out_of_distribution() -> dict | None:
    summary = _read("ood_summary.json")
    if not summary:
        return None

    cells = []
    for cell in summary.get("cells", []):
        deployed = cell.get("deployed", {})
        refit = cell.get("refit_on_benchmark", {})
        cells.append({
            "benchmark": cell.get("benchmark"),
            "arm": cell.get("arm"),
            "label": ARMS.get(cell.get("arm"), cell.get("arm")),
            "auroc": cell.get("auroc_mean_pooled"),
            "deployed_fpr": deployed.get("fpr"),
            "deployed_fnr": deployed.get("fnr"),
            "fnr_at_budget": refit.get("fnr"),
            "ece": cell.get("ece"),
            "n": cell.get("n_documents"),
        })

    bootstrap, mcnemar = [], _read("ood_mcnemar.json") or {}
    for benchmark in BENCHMARKS:
        record = _read(f"ood_significance_{benchmark}.json")
        if not record:
            continue
        paired = mcnemar.get(benchmark, {})
        bootstrap.append({
            "benchmark": benchmark,
            "delta_auroc": record.get("observed_difference"),
            "ci95": record.get("ci95"),
            "reversal": record.get("fraction_of_resamples_where_the_gap_reverses"),
            "discordant": paired.get("discordant"),
            "mcnemar_p": (paired.get("mcnemar") or {}).get("exact_p"),
            "budget": paired.get("budget"),
        })
    return {"cells": cells, "significance": bootstrap}


def _image_probe() -> dict | None:
    record = _read("image_detector_polarity.json")
    if not record:
        return None
    return {
        "model_id": record.get("model_id"),
        "labels": record.get("labels"),
        "n": record.get("n"),
        "median": record.get("median_probability_as_scored"),
        "mean": record.get("mean_probability_as_scored"),
        "separation": record.get("separation"),
        "median_separation": record.get("median_separation"),
        "threshold": record.get("threshold_ai"),
        "human_ceiling": record.get("human_ceiling"),
        "recall": record.get("ai_recall_at_threshold"),
        "in_sample": record.get("operating_point_is_in_sample"),
        "inverted": record.get("inverted_relative_to_labels"),
    }


def build() -> dict:
    """Everything the results page shows. Absent records give absent sections."""
    return {
        "in_distribution": _in_distribution(),
        "out_of_distribution": _out_of_distribution(),
        "image_probe": _image_probe(),
        "source": str(ROOT),
        "note": (
            "Read from the run records committed in reports/experiments/. Nothing is "
            "recomputed here and nothing is hardcoded, so this page cannot drift from the "
            "repository."
        ),
    }
