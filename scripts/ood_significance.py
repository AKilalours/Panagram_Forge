"""Is the out-of-distribution AUROC gap real? A paired bootstrap over the same documents.

WHY PAIRED. Both arms score the SAME benchmark documents, so their errors are correlated:
a document that is hard for one is usually hard for the other. Treating the two AUROCs as
independent samples inflates the standard error and throws away the pairing, which is the
strongest information available. The bootstrap resamples DOCUMENTS and recomputes both
arms' AUROC on the same resample, which preserves it.

WHY THIS MATTERS HERE. The in-distribution comparison gave p = 0.217 and could not
separate the arms. If the out-of-distribution gap is real, this is the number that says so,
and it has to be computed the same careful way rather than asserted because the difference
looks big.

Reports the observed difference, a 95% interval, and the fraction of resamples in which the
gap reverses. CPU only, seconds.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from forge.evaluation import metrics as M


def paired_bootstrap(labels, a_scores, b_scores, n: int = 10_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    size = len(labels)
    observed = float(M.auroc(labels, b_scores) - M.auroc(labels, a_scores))
    diffs = np.empty(n)
    drawn = 0
    for i in range(n):
        # Resample documents, not scores: the pairing lives in the document index.
        idx = rng.integers(0, size, size)
        y = labels[idx]
        if y.min() == y.max():          # a resample with one class has no AUROC
            diffs[i] = np.nan
            continue
        diffs[i] = M.auroc(y, b_scores[idx]) - M.auroc(y, a_scores[idx])
        drawn += 1
    diffs = diffs[~np.isnan(diffs)]
    return {
        "observed_difference": round(observed, 4),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                 round(float(np.percentile(diffs, 97.5)), 4)],
        "fraction_of_resamples_where_the_gap_reverses": round(float((diffs <= 0).mean()), 5),
        "resamples": int(len(diffs)),
        "note": (
            "paired: both arms are scored on the same resampled documents, because they "
            "score the same benchmark and their errors are correlated"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dir", default="reports/experiments")
    args = parser.parse_args()

    root = pathlib.Path(args.dir)
    a = np.load(root / f"ood_scores_baseline_{args.benchmark}.npz")
    b = np.load(root / f"ood_scores_mirror_{args.benchmark}.npz")
    if not np.array_equal(a["labels"], b["labels"]):
        raise SystemExit(
            "the two arms did not score the same documents in the same order, so a paired "
            "test is invalid. Re-run both arms on this benchmark."
        )

    result = paired_bootstrap(a["labels"], a["mean"], b["mean"])
    result["benchmark"] = args.benchmark
    result["auroc"] = {
        "random": round(float(M.auroc(a["labels"], a["mean"])), 4),
        "mirror": round(float(M.auroc(b["labels"], b["mean"])), 4),
    }
    print(json.dumps(result, indent=2))
    (root / f"ood_significance_{args.benchmark}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
