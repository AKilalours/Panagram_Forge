"""McNemar on the same documents, at a MATCHED false-positive budget.

Why this exists, and why `ood_table.py` no longer decides significance on its own.

The first version of the significance step ran a two-proportion z-test on each arm's FNR
at *its own* deployed threshold. Two things were wrong with that, and both inflated the
verdict:

1.  The two arms sat at different operating points. On HC3 the control arm spent 0.55% of
    the human FPR budget and the mirror arm spent 2.4%. An arm allowed to be four times
    more trigger-happy misses less AI text for that reason alone, so part of the gap being
    tested was the threshold, not the model.
2.  The z-test assumes two independent samples. Both arms score the *identical* documents,
    so their errors are correlated and the unpaired standard error is the wrong denominator.

The consequence was a file asserting `significant_at_0.05: true` on all three benchmarks.
Re-run correctly, MAGE is not significant (p = 0.24). That verdict was an artefact.

This script fixes both: each arm's threshold is re-fit on the benchmark so that both spend
the same human false-positive budget, and the test is McNemar's on the discordant AI
documents, which is the paired test this design calls for.

The re-fit threshold is optimistic and unavailable at deployment. It is used here only so
that the arms are compared at the same operating point rather than at two different ones.
The deployed-threshold numbers stay in the per-cell files as description, not as a test.

Run after `eval_ood.py` has written the per-document score arrays. CPU only.
"""

from __future__ import annotations

import glob
import json
import math
import pathlib

import numpy as np

BENCHMARKS = ("hc3", "mage", "raid")
ARMS = {"random": "baseline", "mirror": "mirror"}
BUDGET = 0.001
OUT = pathlib.Path("reports/experiments/ood_mcnemar.json")

TEST_NOTE = (
    "McNemar on the same AI documents, at a threshold re-fit per arm to spend the SAME "
    "human false-positive budget. Replaces the two-proportion z-test, which compared FNRs "
    "at two different operating points and assumed the two arms were independent samples "
    "when they score identical documents."
)


def _load(arm_dir: str, benchmark: str):
    path = pathlib.Path(f"reports/experiments/ood_scores_{arm_dir}_{benchmark}.npz")
    if not path.exists():
        return None
    d = np.load(path)
    missing = {"labels", "mean"} - set(d.files)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}, has {sorted(d.files)}")
    return d["labels"].astype(int), d["mean"].astype(float)


def threshold_at_budget(human_scores: np.ndarray, budget: float) -> tuple[float, int]:
    """Highest threshold that lets at most floor(budget * n) human documents through.

    Returns the threshold and the number of false positives it permits. With 2,000 human
    documents and a 0.1% budget that allowance is TWO documents, so the threshold is the
    third-largest human score and the resolution of this comparison is coarse. That
    coarseness is the honest limit of a 2,000-document benchmark, not a bug to smooth over.
    """
    allowed = int(math.floor(budget * human_scores.size))
    if allowed >= human_scores.size:
        raise ValueError("budget admits every human document")
    return float(np.sort(human_scores)[::-1][allowed]), allowed


def mcnemar(discordant_b: int, discordant_c: int) -> dict:
    """Continuity-corrected chi-square plus the exact binomial, which rules at low counts."""
    n = discordant_b + discordant_c
    if n == 0:
        return {"chi2": None, "p": None, "exact_p": 1.0, "note": "no discordant documents"}
    chi2 = (abs(discordant_b - discordant_c) - 1) ** 2 / n
    k = min(discordant_b, discordant_c)
    exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {
        "chi2": round(chi2, 3),
        "p": float(f"{math.erfc(math.sqrt(chi2 / 2)):.6g}"),
        "exact_p": float(f"{exact:.6g}"),
    }


def main() -> int:
    if not glob.glob("reports/experiments/ood_scores_*.npz"):
        print("no per-document score arrays; run eval_ood.py first")
        return 2

    out: dict[str, dict] = {}
    for benchmark in BENCHMARKS:
        loaded = {name: _load(d, benchmark) for name, d in ARMS.items()}
        if any(v is None for v in loaded.values()):
            print(f"{benchmark}: not run for both arms, skipped")
            continue
        (lab_r, s_r), (lab_m, s_m) = loaded["random"], loaded["mirror"]
        if not np.array_equal(lab_r, lab_m):
            raise RuntimeError(
                f"{benchmark}: the arms scored different documents, or the same documents "
                "in a different order. Every paired statistic here would be meaningless."
            )

        human = lab_r == 0
        ai = lab_r == 1
        thr_r, allowed = threshold_at_budget(s_r[human], BUDGET)
        thr_m, _ = threshold_at_budget(s_m[human], BUDGET)

        miss_r = s_r[ai] <= thr_r
        miss_m = s_m[ai] <= thr_m
        b = int((miss_r & ~miss_m).sum())
        c = int((~miss_r & miss_m).sum())

        out[benchmark] = {
            "budget": BUDGET,
            "allowed_human_fp": allowed,
            "threshold": {"random": thr_r, "mirror": thr_m},
            "human_fp_spent": {
                "random": int((s_r[human] > thr_r).sum()),
                "mirror": int((s_m[human] > thr_m).sum()),
            },
            "n_ai": int(ai.sum()),
            "ai_misses": {"random": int(miss_r.sum()), "mirror": int(miss_m.sum())},
            "discordant": {"random_miss_mirror_catch": b, "random_catch_mirror_miss": c},
            "concordant": {
                "both_miss": int((miss_r & miss_m).sum()),
                "both_catch": int((~miss_r & ~miss_m).sum()),
            },
            "mcnemar": mcnemar(b, c),
            "test": TEST_NOTE,
        }
        r = out[benchmark]
        print(
            f"{benchmark:<5} misses {r['ai_misses']['random']:>5} / "
            f"{r['ai_misses']['mirror']:<5} discordant {b:>4} vs {c:<4} "
            f"exact p = {r['mcnemar']['exact_p']:.4g}"
        )

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
