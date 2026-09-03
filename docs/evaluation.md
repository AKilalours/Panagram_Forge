# Evaluation

The ablation table. No cell is filled until the run exists and its `dataset_version`
and `code_commit` are recorded in `reports/experiments/`.

| System | Human FPR | AI FNR | OOD AUROC | Adversarial FNR | ECE |
|---|---|---|---|---|---|
| A: random synthetic | 0.000502 | 0.00719 | not run | not run | 0.00447 |
| B: + synthetic mirrors | 0.000502 | 0.00430 | not run | not run | 0.00371 |
| C: + hard negatives | | | | | |
| D: + adversarial | | | | | |

Arms A and B, `dataset_version` v0.1-min, `code_commit` 4ec8204, 2026-09-03. Full records
in `reports/experiments/`. Both arms: 20,000 human and 20,000 AI documents, two epochs,
DeBERTa-v3-base, one RTX 4090, about 21 minutes each.

## The result, and why it is not yet a finding

At an identical false-positive budget the mirror arm missed fewer AI documents:

| | random | mirror |
|---|---|---|
| False positives | 1 of 1,993 humans | 1 of 1,993 humans |
| Missed AI documents | **15 of 2,086** | **9 of 2,091** |
| FNR | 0.719% | 0.430% |
| Relative reduction | | 40.1% |

**That difference is not statistically significant.** A two-proportion z-test on the FNR
gives z = 1.234, two-sided p = 0.217. Detecting a gap this size at 80% power would need
about **10,761 AI documents per arm in validation**; there are 2,090. The 40% headline rests
on the difference between nine misses and fifteen, and six documents is well inside what
chance produces.

Reported this way deliberately. The number that sells is 40%; the number that is true is
p = 0.217, and the gap between those two sentences is the reason this section exists.

**The validation split is also saturated.** AUROC is 0.99997 on both arms. The detector is
scoring text from the same four generator families it was trained on, and that task is
close to solved at this scale. An in-distribution comparison between two nearly perfect
models cannot separate them.

**So the honest reading is:** the pipeline runs end to end, both arms hit the 0.1% FPR
budget, calibration is good (ECE 0.0045 and 0.0037), and the mirror arm points the right
way without proving anything. The discriminating test is held-out generators, RAID, MAGE
and HC3, which has not run.

## What made these numbers trustworthy enough to report at all

Before this run, a detector could score **AUROC 0.841 on the control arm without reading a
word of the text**, purely from document length. Generation used one `max_new_tokens` for
every document and the models treated the cap as a target, so AI text ran systematically
long. Two corpora were discarded rather than trained on.

| stage | length-only AUROC (random / mirror) |
|---|---|
| Original corpora | 0.841 / 0.774 |
| Per-document token budget | 0.720 / 0.699 |
| AI length matched to the human corpus | **0.546 / 0.546** |

The final assembled training sets have a human median of 256 words against an AI median of
257. A gate now blocks training whenever length separates the classes by more than 0.15
from chance, measured on the assembled data rather than the raw pools.

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

**Length overshoot dominated mirror rejection, and it was structural.** The first real
run accepted 18,856 of 30,000 mirrors, a 31.5% rate over 59,920 attempts. The reasons:

| reason | count |
|---|---|
| too_long | 33,912 |
| too_short | 6,070 |
| assistant_preamble | 999 |
| empty | 83 |

Length overshoot is 85% of all rejections and the cause is structural rather than random:
generation used one `max_new_tokens` of 640 for every document, regardless of its target
length. A source with a 250-token target could never be matched, because the model writes
to the cap and lands at a ratio of about 2.6 against a ceiling of 1.6. That also explains
why retries recovered so little: the seed changes between attempts, the length ceiling does
not. Attempt 1 recovered 632 of 3,055 failures on the first family.

Consequence, and the reason the arms are capped by distribution rather than by count: the
mirrors that survive skew long, while the control arm draws its target lengths from the
whole human corpus and does not skew. Capping both arms to an equal count alone would leave
them differing in length distribution as well as in matching, and a detector can learn
length. Both arms are therefore capped with `cap_documents_matching`, which selects the
control arm's documents so its length profile tracks the mirror arm's.

Fix for the next run, not applied here: set `max_new_tokens` per document from its target
rather than globally. Applying it now would mean regenerating both arms.

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
