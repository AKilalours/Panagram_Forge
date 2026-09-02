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

## Recorded conditions for the first run (Tier 1, arms A and B)

Written before any result exists, so that the caveats are not chosen after seeing which
way the numbers went.

**Scale.** 60,000 human documents; 30,000 AI documents per arm. Reduced from the planned
60,000 per arm to fit a single-GPU compute budget. The budget is equal across arms, which
is what the comparison depends on; what is lost is statistical precision, not validity.

**Class balance.** Roughly 2:1 human to AI in training. Identical for both arms, so the
comparison is unaffected, but absolute FPR and FNR are shifted relative to a balanced set
and must not be quoted as if they came from one.

**Generator scale.** Four held-in families at 1.7B to 3.8B parameters. Smaller models
produce more detectable text, so absolute FNR will look better than a 7B-to-70B roster
would give. Only the difference between arms is meaningful.

**Validator asymmetry between arms.** The mirror arm accepts a generated document only if
its length ratio to the source falls in [0.6, 1.6]; the random arm uses [0.5, 2.0] against
a target length drawn from the corpus distribution. The thresholds differ because matching
is the point of a mirror and there is nothing to match in the control. The consequence is
that arm B's data passed a stricter filter than arm A's. If arm B wins, this is the first
alternative explanation a reader will propose, and it cannot be ruled out from these two
runs alone. Ruling it out needs an arm B variant generated under the looser thresholds.

**Selection pressure from rejection.** First-pass rejection on the mirror arm ran near 40%
for the first generator family. Documents are retried up to twice with different seeds, so
final acceptance is higher, but the accepted set is selected rather than sampled: it
over-represents source documents whose mirrors happened to land in the accepted length
band. The per-reason breakdown is recorded in `reports/experiments/`.

**Single GPU.** One RTX 4090. The FSDP and DeepSpeed paths exist and are tested but were
not exercised for these runs, so no multi-GPU scaling claim is made.

**Not measured here.** Arms C and D, external benchmarks (RAID, MAGE, HC3), the adversarial
suite, and calibration under distribution shift. Those are Tier 2 and their rows stay empty
until they are run.
