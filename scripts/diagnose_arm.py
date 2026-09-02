"""Explain a suspicious smoke-run metric before paying for a training run.

WHY THIS EXISTS. The first minimal smoke run finished cleanly and reported

    auroc 0.1185   fnr 1.0   fpr 0.0   temperature 10.0   n_human 35   n_ai 41

AUROC 0.118 on 76 examples is roughly five standard errors BELOW chance. An undertrained
model scores about 0.5; a model that scores 0.12 is ranking the classes backwards, and that
has three possible causes with very different consequences:

  1. LABEL POLARITY. Something between the loader and the metric flips the classes. This
     would invalidate every number the project produces. Cheapest to rule out, so first.

  2. A LENGTH SHORTCUT IN THE OPPOSITE DIRECTION. Generation used one max_new_tokens for
     every document, so AI documents are bounded in length while human documents are not.
     Mean-pooled embeddings correlate with length, and a randomly initialised linear head
     on top of them ranks by length for free. If AI documents are systematically shorter,
     an untrained model scores below chance BEFORE it has learned anything. That is not a
     bug in the loop; it is the confound this project exists to study, showing up early.

  3. THE MODEL ACTUALLY LEARNED AN INVERSION in three optimizer steps, which would be
     surprising and would point at inconsistent labels between train and val.

The discriminating measurement is the UNTRAINED model. If a freshly built model scores the
same 0.12 on the same split, then training inverted nothing and cause 3 is out. If the
untrained model scores 0.5 and the trained one scores 0.12, three optimizer steps did that,
and something is wrong with the data.

This script does not train. Run it before any paid run.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from forge.common.config import load


def _words(text: str) -> int:
    return len(text.split())


def main(config_path: str) -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.data import Collator, build_dataset, load_examples
    from forge.training.train import _reference_arm, evaluate_split, validate_config

    cfg = load(config_path)
    mcfg = load(cfg["model_config"])
    dcfg, paths = cfg["data"], cfg.get("paths", {})
    arm = validate_config(cfg)

    examples = load_examples(
        human_root=paths.get("human"),
        ai_root=paths.get("ai") or paths.get("mirror"),
        mixed_root=paths.get("mixed"),
        limit=200,
        expect_arm=arm,
        ai_cap=dcfg.get("ai_cap"),
        ai_reference=_reference_arm(dcfg),
    )

    # ---------------------------------------------------------------- 1. labels and lengths
    print(f"\narm: {arm}   examples: {len(examples)}")
    print("label counts:", dict(Counter(e.label for e in examples)))

    lengths: dict[int, list[int]] = {0: [], 1: []}
    for e in examples:
        lengths[int(e.label)].append(_words(e.text))
    for label, name in ((0, "human"), (1, "ai   ")):
        arr = np.array(lengths[label] or [0])
        print(
            f"{name} n={len(arr):>4}  words: median {np.median(arr):>6.0f}  "
            f"mean {arr.mean():>6.0f}  p10 {np.percentile(arr, 10):>6.0f}  "
            f"p90 {np.percentile(arr, 90):>6.0f}"
        )

    # A single number for how separable the classes are by LENGTH ALONE. This is the score
    # any model gets for free, without reading a word of the text.
    y = np.array([int(e.label) for e in examples])
    n = np.array([_words(e.text) for e in examples], dtype=float)
    from forge.evaluation import metrics as M

    print(f"\nAUROC from LENGTH ALONE (longer = more AI): {M.auroc(y, n):.4f}")
    print(f"AUROC from LENGTH ALONE (shorter = more AI): {M.auroc(y, -n):.4f}")
    print("  0.5 means length carries no signal. Far from 0.5 means a detector can score")
    print("  well without reading the text, and the arms must be length matched.")

    # ------------------------------------------------------- 2. the untrained model's score
    tok = AutoTokenizer.from_pretrained(mcfg["backbone"])
    feats = build_dataset(examples, tok, max_length=mcfg["max_length"],
                          stride=mcfg["window"]["stride"])
    val = [f for f in feats if f["split"] == "val"] or [f for f in feats if f["split"] == "train"]
    coll = Collator(tok, max_length=mcfg["max_length"])
    loader = DataLoader(val, batch_size=16, shuffle=False, collate_fn=coll)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["training"].get("seed", 42))
    model = build_model(ForgeConfig(
        backbone=mcfg["backbone"], max_length=mcfg["max_length"],
        stride=mcfg["window"]["stride"],
    )).to(device)

    result = evaluate_split(model, loader, device, cfg.get("fpr_budget", 0.001))
    print(f"\nUNTRAINED model on the same val split: auroc {result.auroc:.4f} "
          f"(n_human {result.n_human}, n_ai {result.n_ai})")
    print("  Compare against the smoke run's auroc. If they are close, training inverted")
    print("  nothing and the ranking comes from initialisation plus length. If the")
    print("  untrained model sits near 0.5 and the trained one does not, the labels")
    print("  disagree between train and val and nothing should be run until that is found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "configs/training/baseline_minimal.yaml"))
