"""Collect the out-of-distribution cells into one table, with the significance test.

Written as a separate step so the table can be regenerated from the JSON without rerunning
a single GPU hour, and so the significance arithmetic lives in one place rather than being
retyped into prose.
"""

from __future__ import annotations

import json
import math
import pathlib

BENCHMARKS = ("hc3", "mage", "raid")
ARMS = ("baseline", "mirror")
LABEL = {"baseline": "A: random", "mirror": "B: mirrors"}


def _load(arm: str, benchmark: str) -> dict | None:
    path = pathlib.Path(f"reports/experiments/ood_{arm}_{benchmark}.json")
    return json.loads(path.read_text()) if path.exists() else None


def _significance(a: dict, b: dict) -> dict:
    """Two-proportion z-test on the FNR at the deployed threshold.

    The same test the in-distribution comparison failed. Applied here before anyone reads
    the direction of the gap, so the answer is not chosen after seeing which way it went.
    """
    n_a, n_b = a["n_ai"], b["n_ai"]
    p_a, p_b = a["deployed"]["fnr"], b["deployed"]["fnr"]
    miss_a, miss_b = round(p_a * n_a), round(p_b * n_b)
    if not n_a or not n_b:
        return {"error": "no AI documents"}
    pooled = (miss_a + miss_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b)) if pooled else 0.0
    if se == 0:
        return {"misses": [miss_a, miss_b], "note": "no misses in either arm"}
    z = (p_a - p_b) / se
    return {
        "misses": {"random": miss_a, "mirror": miss_b},
        "n_ai": {"random": n_a, "mirror": n_b},
        "z": round(z, 3),
        "p_two_sided": round(math.erfc(abs(z) / math.sqrt(2)), 4),
        "significant_at_0.05": bool(math.erfc(abs(z) / math.sqrt(2)) < 0.05),
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
            test = _significance(cells["baseline"], cells["mirror"])
            summary[benchmark] = test
            print(f"{'':8} significance: {test}")
    pathlib.Path("reports/experiments/ood_summary.json").write_text(
        json.dumps(
            {
                "note": (
                    "FPR and FNR at the DEPLOYED threshold, fit on each arm's validation "
                    "split. Re-fit values are in the per-cell files and are optimistic."
                ),
                "cells": rows,
                "significance": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print("\nwritten reports/experiments/ood_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
