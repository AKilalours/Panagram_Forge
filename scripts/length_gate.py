"""Measure the length confound on the data the model will ACTUALLY be trained on.

WHY THIS FILE EXISTS. The gate inside run_all.sh loaded the raw arm directories: no
ai_cap, no human_cap, no length matching. It therefore measured a corpus that no training
run would ever see, and refused to train on a dataset that had not been assembled yet.

That is the project's recurring failure in its purest form: a check that reports a real
number about the wrong thing. The first version of this gate stopped a run for a confound
that the loader was about to remove.

This loads through `load_examples` with the arm's own config, exactly as
`forge.training.train.run` does, so the number describes the assembled training set.

Exit code 0 means the gate passes, 2 means it fails, and the threshold is an argument so
the caller decides policy rather than this script.
"""

from __future__ import annotations

import sys

import numpy as np

from forge.common.config import load
from forge.evaluation import metrics as M
from forge.training.data import load_examples
from forge.training.train import REFERENCE_HUMAN, _reference_arm, validate_config

ARMS = ("configs/training/mirror_minimal.yaml", "configs/training/baseline_minimal.yaml")


def deviation(config_path: str) -> float:
    cfg = load(config_path)
    dcfg, paths = cfg["data"], cfg.get("paths", {})
    arm = validate_config(cfg)

    human_rows = (
        load_examples(human_root=paths.get("human"))
        if dcfg.get("ai_reference") == REFERENCE_HUMAN
        else None
    )
    rows = load_examples(
        human_root=paths.get("human"),
        ai_root=paths.get("ai") or paths.get("mirror"),
        limit=None,
        expect_arm=arm,
        ai_cap=dcfg.get("ai_cap"),
        ai_reference=_reference_arm(dcfg, human_rows=human_rows),
        human_cap=dcfg.get("human_cap"),
    )

    labels = np.array([int(r.label) for r in rows])
    words = np.array([len(r.text.split()) for r in rows], dtype=float)
    auroc = float(M.auroc(labels, words))

    human = words[labels == 0]
    ai = words[labels == 1]
    print(
        f"{arm:>9}  human n={len(human):>6} median={np.median(human):>5.0f}  "
        f"ai n={len(ai):>6} median={np.median(ai):>5.0f}  "
        f"length-only AUROC={auroc:.4f}"
    )
    return abs(auroc - 0.5)


def main() -> int:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    worst = max(deviation(path) for path in ARMS)
    print(f"worst deviation from chance: {worst:.4f}  (threshold {threshold})")
    if worst > threshold:
        print(
            "GATE FAILED. Length still separates the classes on the assembled training "
            "set, so a detector could reach that score without reading a word."
        )
        return 2
    print("GATE PASSED. Length carries little signal on the data that will be trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
