"""Collect the out-of-distribution cells into one table.

Written as a separate step so the table can be regenerated from the JSON without rerunning
a single GPU hour.

This step reports and does not adjudicate. The significance tests live in ood_mcnemar.py
(paired, matched budget) and ood_significance.py (paired bootstrap on AUROC), because the
numbers in this table come from two different operating points and cannot support a test.
"""

from __future__ import annotations

import json
import pathlib

BENCHMARKS = ("hc3", "mage", "raid")
ARMS = ("baseline", "mirror")
LABEL = {"baseline": "A: random", "mirror": "B: mirrors"}


def _load(arm: str, benchmark: str) -> dict | None:
    path = pathlib.Path(f"reports/experiments/ood_{arm}_{benchmark}.json")
    return json.loads(path.read_text()) if path.exists() else None


def _deployed_description(a: dict, b: dict) -> dict:
    """Describe the deployed-threshold miss counts. Deliberately returns NO verdict.

    This function used to run a two-proportion z-test here and emit
    `significant_at_0.05`. That was wrong twice over, and both errors pushed the answer
    toward "significant":

      1. The two arms sit at DIFFERENT operating points. Each `deployed` threshold is fit
         on that arm's own validation split, and the FPR they actually spend on a benchmark
         differs, 0.55% against 2.4% on HC3. An arm permitted four times the false-positive
         rate misses less AI text for that reason alone. Comparing FNR across unequal FPR
         is not a comparison of detectors.
      2. The z-test assumes independent samples. Both arms score the IDENTICAL documents,
         so the errors are correlated and the unpaired standard error is too large a
         denominator to be conservative in the direction anyone would want.

    The file it wrote claimed significance on all three benchmarks. Re-tested properly with
    McNemar at a matched budget, MAGE is not significant (p = 0.24). The verdict was an
    artefact of the test, not a finding.

    Significance now lives in two places that can support it:
      - `scripts/ood_mcnemar.py`, paired McNemar at a matched false-positive budget
      - `scripts/ood_significance.py`, paired bootstrap on AUROC

    What stays here is description: how many documents each arm missed, and at what cost
    in false positives, so the two are never read without each other.
    """
    n_a, n_b = a["n_ai"], b["n_ai"]
    if not n_a or not n_b:
        return {"error": "no AI documents"}
    da, db = a["deployed"], b["deployed"]
    return {
        "misses": {"random": round(da["fnr"] * n_a), "mirror": round(db["fnr"] * n_b)},
        "n_ai": {"random": n_a, "mirror": n_b},
        "fpr_spent": {"random": da["fpr"], "mirror": db["fpr"]},
        "thresholds_are_matched": bool(da["fpr"] == db["fpr"]),
        "no_test_here": (
            "The arms sit at different false-positive rates, so these two miss counts are "
            "NOT comparable and no test is run on them. See ood_mcnemar.json for the "
            "matched-budget paired test and ood_significance_*.json for the AUROC bootstrap."
        ),
    }


def main() -> int:
    rows, summary = [], {}
    print(f"{'benchmark':<8} {'arm':<11} {'n':>6} {'AUROC':>8} {'FPR':>8} {'FNR':>8}")
    print("-" * 54)
    for benchmark in BENCHMARKS:
        cells = {arm: _load(arm, benchmark) for arm in ARMS}
        for arm in ARMS:
            cell = cells[arm]
            if not cell:
                print(f"{benchmark:<8} {LABEL[arm]:<11} {'not run':>6}")
                continue
            d = cell["deployed"]
            print(
                f"{benchmark:<8} {LABEL[arm]:<11} {cell['n_documents']:>6} "
                f"{cell['auroc_mean_pooled']:>8.4f} {d['fpr']:>8.4f} {d['fnr']:>8.4f}"
            )
            rows.append(cell)
        if cells["baseline"] and cells["mirror"]:
            described = _deployed_description(cells["baseline"], cells["mirror"])
            summary[benchmark] = described
            print(f"{'':8} deployed misses (NOT a test): {described['misses']} "
                  f"at FPR {described['fpr_spent']}")
    pathlib.Path("reports/experiments/ood_summary.json").write_text(
        json.dumps(
            {
                "note": (
                    "FPR and FNR at the DEPLOYED threshold, fit on each arm's validation "
                    "split. Re-fit values are in the per-cell files and are optimistic."
                ),
                "cells": rows,
                "deployed_threshold_description": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print("\nwritten reports/experiments/ood_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
