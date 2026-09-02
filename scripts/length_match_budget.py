"""How large can an arm be if its AI documents match the HUMAN length distribution?

THE PROBLEM THIS ANSWERS. Both arms carry a length shortcut, measured on the real data:

    baseline (random)  AI median 394 words vs human 250   AUROC from length alone 0.854
    mirror             AI median 364 words vs human 250   AUROC from length alone 0.776

A detector reaches 0.85 on the baseline arm without reading a word. Both arms would train a
length detector, the mirror-versus-random comparison would compress toward the difference
between 0.854 and 0.776, and the headline claim would have an obvious alternative
explanation that a reviewer would find in one plot.

WHY THE EXISTING MATCHING DOES NOT FIX IT. `cap_documents_matching` reshapes each arm's AI
pool to match the OTHER ARM's AI pool. Both are long. It makes the arms comparable to each
other, which is what it was written for, and does nothing about the human-versus-AI gap
inside either arm.

THE TENSION THIS SCRIPT EXISTS TO QUANTIFY. Matching the AI pool to the human pool means
discarding AI documents from over-represented length cells. You cannot hold both an exact
budget and an exact matched distribution when the pools have different shapes: the existing
second pass keeps the count exact by refilling from whatever cells still have documents,
which quietly puts the skew back. So the honest move is to find the largest budget at which
the match needs no refill, and use that.

For each arm this reports that number. The smaller of the two becomes the shared budget, the
same way the raw counts did.

WHAT TO DO WITH THE ANSWER. If the matched budget is a workable fraction of the corpus, the
fix is data-side, deterministic, and costs no GPU. If it is tiny, the pools barely overlap in
length and the real fix is regeneration with max_new_tokens conditioned on each source
document, which costs GPU hours and should be decided deliberately rather than discovered
halfway through a run.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from forge.common.config import load
from forge.evaluation import metrics as M
from forge.training.data import LENGTH_BINS, length_bin, load_examples
from forge.training.train import validate_config

ARMS = ("configs/training/baseline_minimal.yaml", "configs/training/mirror_minimal.yaml")


def _cells(rows) -> Counter:
    return Counter((r.split, length_bin(r.text)) for r in rows)


def _matched_budget(human, ai) -> tuple[int, list[str]]:
    """Largest N such that every (split, length bin) cell can supply its human-shaped share.

    Bounded by the SCARCEST cell, which is the point: one cell that the AI pool cannot fill
    caps the whole matched sample, and the report names it so the cause is visible rather
    than inferred from a small number.
    """
    human_cells, ai_cells = _cells(human), _cells(ai)
    total = sum(human_cells.values()) or 1

    budget = None
    binding: list[tuple[float, str]] = []
    for cell, count in human_cells.items():
        share = count / total
        available = ai_cells.get(cell, 0)
        limit = int(available / share) if share else None
        if limit is not None:
            binding.append((limit, f"{cell[0]:>5} {cell[1]:>5}w  needs {share:6.2%}  has {available:>6}"))
            budget = limit if budget is None else min(budget, limit)
    binding.sort()
    return (budget or 0), [line for _, line in binding[:5]]


def _length_auroc(human, ai) -> float:
    y = np.array([0] * len(human) + [1] * len(ai))
    n = np.array([len(r.text.split()) for r in human + ai], dtype=float)
    return float(M.auroc(y, n))


def report(config_path: str) -> int:
    cfg = load(config_path)
    paths = cfg.get("paths", {})
    arm = validate_config(cfg)

    rows = load_examples(
        human_root=paths.get("human"),
        ai_root=paths.get("ai") or paths.get("mirror"),
        limit=None,
        expect_arm=arm,
    )
    human = [r for r in rows if r.label == 0]
    ai = [r for r in rows if r.label == 1]

    print(f"\n=== {arm}  ({config_path})")
    print(f"human {len(human):>7}   ai {len(ai):>7}")
    print(f"AUROC from length alone, unmatched: {_length_auroc(human, ai):.4f}")

    budget, binding = _matched_budget(human, ai)
    print(f"matched budget: {budget} AI documents "
          f"({budget / max(len(ai), 1):.1%} of this arm's AI pool)")
    print("scarcest cells, which are what cap it:")
    for line in binding:
        print("   ", line)
    return budget


def main() -> int:
    print("LENGTH BINS (words):", LENGTH_BINS)
    budgets = {path: report(path) for path in (sys.argv[1:] or ARMS)}
    smallest = min(budgets.values())
    print("\n" + "=" * 70)
    for path, budget in budgets.items():
        print(f"{budget:>8}  {path}")
    print(f"\nshared matched budget would be {smallest}")
    print("Compare against the current ai_cap of 18856. If this is a workable fraction of")
    print("that, the confound is removable on the CPU today. If it is a small fraction, the")
    print("pools barely overlap in length and the real fix is regeneration with")
    print("max_new_tokens conditioned on each source document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
