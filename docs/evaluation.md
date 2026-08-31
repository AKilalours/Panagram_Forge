# Evaluation

The ablation table. No cell is filled until the run exists and its `dataset_version`
and `code_commit` are recorded in `reports/experiments/`.

| System | Human FPR | AI FNR | OOD AUROC | Adversarial FNR | ECE |
|---|---|---|---|---|---|
| A: random synthetic | | | | | |
| B: + synthetic mirrors | | | | | |
| C: + hard negatives | | | | | |
| D: + adversarial | | | | | |

Data budget is held equal across A, B and C. Without that, any improvement is just
"more data" and says nothing about failure-driven selection.

## What counts as a result

A positive result: failure-driven mirroring reduces OOD error by a measured amount
using fewer additional synthetic samples than random augmentation, at equal or better
human FPR.

A negative result is still a result and gets written up as one, for example:
"failure-driven selection improved human FPR but did not improve unseen-generator
generalization." What is not acceptable is filling this table with numbers that were
not measured.
