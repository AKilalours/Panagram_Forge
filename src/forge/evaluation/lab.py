"""The evaluation lab.

Design rules, each of which exists because the obvious alternative produces a better
looking but meaningless number.

**The threshold is fitted on validation, never on the evaluation set.** It is chosen to
hit the FPR budget on held-out HUMAN documents, then applied unchanged everywhere.
Re-picking the threshold per regime would report the best case each time and hide the
fact that a single deployed threshold cannot satisfy all of them at once. That single
threshold is what production actually runs.

**Every regime reports FPR first.** A regime where AI detection improves while FPR rises
is a regression, not a win.

**External benchmarks are contamination-checked before they are scored.** A single
overlapping document between training and RAID invalidates the external claim, and
overlap does not raise an error on its own.

**A regime with too few human documents to measure the FPR budget reports that**, rather
than reporting an FPR of 0.0. At a 0.001 budget you need on the order of thousands of
human examples before the number means anything, and 0/50 is not evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forge.evaluation import metrics as M


@dataclass
class RegimeScores:
    name: str
    n: int
    n_human: int
    n_ai: int
    fpr: float
    fnr: float
    auroc: float
    ece: float
    precision: float
    fpr_measurable: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "regime": self.name, "n": self.n, "n_human": self.n_human, "n_ai": self.n_ai,
            "fpr": _r(self.fpr), "fnr": _r(self.fnr), "auroc": _r(self.auroc),
            "ece": _r(self.ece), "precision": _r(self.precision),
            "fpr_measurable": self.fpr_measurable, "note": self.note,
        }


def _r(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def min_humans_for_fpr(budget: float, min_expected_errors: int = 5) -> int:
    """Rule of thumb: to distinguish an FPR of `budget` from zero you want to expect at
    least a handful of false positives. Below that, a reported 0.0 is noise."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    return int(np.ceil(min_expected_errors / budget))


def precision_at(y_true, y_score, threshold: float) -> float:
    y_true = np.asarray(y_true, int)
    pred = np.asarray(y_score, float) >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    return tp / (tp + fp) if (tp + fp) else float("nan")


def choose_threshold(val_labels, val_scores, fpr_budget: float) -> float:
    """One threshold, fitted once on validation humans, used everywhere after."""
    return M.threshold_at_fpr(val_labels, val_scores, fpr_budget)


def score_regime(name: str, y_true, y_score, threshold: float, fpr_budget: float) -> RegimeScores:
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    n_human = int((y_true == 0).sum())
    n_ai = int((y_true == 1).sum())
    needed = min_humans_for_fpr(fpr_budget)
    measurable = n_human >= needed
    return RegimeScores(
        name=name, n=len(y_true), n_human=n_human, n_ai=n_ai,
        fpr=M.false_positive_rate(y_true, y_score, threshold),
        fnr=M.false_negative_rate(y_true, y_score, threshold),
        auroc=M.auroc(y_true, y_score),
        ece=M.expected_calibration_error(y_true, y_score),
        precision=precision_at(y_true, y_score, threshold),
        fpr_measurable=measurable,
        note="" if measurable else f"only {n_human} human docs; need ~{needed} to resolve an FPR of {fpr_budget}",
    )


@dataclass
class LabReport:
    model_version: str
    dataset_version: str
    code_commit: str
    threshold: float
    fpr_budget: float
    regimes: list[RegimeScores] = field(default_factory=list)
    contamination: dict = field(default_factory=dict)
    adversarial: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "code_commit": self.code_commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fpr_budget": self.fpr_budget,
            "threshold": _r(self.threshold),
            "regimes": [r.as_dict() for r in self.regimes],
            "contamination": self.contamination,
            "adversarial": self.adversarial,
        }

    def headline(self) -> dict:
        """The row that goes in docs/evaluation.md. Pulled from named regimes so the
        table cannot accidentally be filled from the easiest one."""
        by = {r.name: r for r in self.regimes}
        iid = by.get("R1_iid")
        ood = by.get("R3_unseen_generator")
        adv = by.get("R5_adversarial")
        return {
            "human_fpr": _r(iid.fpr) if iid else None,
            "ai_fnr": _r(iid.fnr) if iid else None,
            "ood_auroc": _r(ood.auroc) if ood else None,
            "adversarial_fnr": _r(adv.fnr) if adv else None,
            "ece": _r(iid.ece) if iid else None,
        }

    def write(self, root: str | Path) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        p = root / f"{self.model_version}_eval.json"
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return p


def run_lab(
    predictions: dict[str, tuple[list[int], list[float]]],
    val: tuple[list[int], list[float]],
    fpr_budget: float,
    model_version: str,
    dataset_version: str,
    code_commit: str,
    train_hashes: set[str] | None = None,
    eval_hashes: dict[str, set[str]] | None = None,
) -> LabReport:
    """predictions maps regime name -> (labels, scores). `val` fits the threshold."""
    threshold = choose_threshold(*val, fpr_budget)
    report = LabReport(
        model_version=model_version, dataset_version=dataset_version,
        code_commit=code_commit, threshold=threshold, fpr_budget=fpr_budget,
    )
    for name, (y, s) in predictions.items():
        report.regimes.append(score_regime(name, y, s, threshold, fpr_budget))

    if train_hashes is not None and eval_hashes:
        for bench, hashes in eval_hashes.items():
            overlap = train_hashes & hashes
            report.contamination[bench] = {
                "overlap": len(overlap),
                "clean": not overlap,
                "examples": sorted(overlap)[:5],
            }
    return report


def adversarial_table(clean_fnr: float, per_attack_fnr: dict[str, float]) -> dict:
    """delta-FNR per attack. Positive means the attack bought the evader something."""
    return {
        atk: {"fnr": _r(f), "delta_fnr": _r(M.robustness_delta(f, clean_fnr))}
        for atk, f in sorted(per_attack_fnr.items())
    }
