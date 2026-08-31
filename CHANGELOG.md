# Changelog

## v0.1.0 - 2026-08-31 - Phase 0

- Repository scaffold, packaging, CI, Docker (API and CUDA training images).
- **Data Specification v1 frozen** (`docs/data_spec_v1.md`), verified against live
  dataset cards for FineWeb, FineWeb-Edu, HC3, RAID and MAGE.
- Implemented and tested: group-aware deterministic splits, detection metrics
  (FPR, FNR, threshold-at-FPR, AUROC, ECE), long-document windowing, release gate
  policy, model registry contract, MAGE label-polarity guard, config spec-check.
- Corrections to the original plan, recorded in the spec:
  - FineWeb `sample-10BT` instead of `sample-100BT` (277 GB is not workable locally).
  - RAID evaluation uses `RAID-extra`, since `RAID-test` labels are held out.
  - RAID `code`, `Czech`, `German` domains excluded from the headline number.
  - MAGE label polarity asserted at load time rather than trusted.

## Phase 2 - mirror engine

- Attribute extraction (heuristic, offline-deterministic) with a hard no-verbatim-copy
  guard; an LLM extractor goes through the same guard.
- Generator backends behind one interface: vLLM, Transformers, API, plus a Fake backend
  so the pipeline is verifiable without a GPU. Unpinned model revisions are refused.
- Deterministic assignment: family and decoding are seeded by the human doc id, so a
  partial run resumes to the same dataset. Held-out families are structurally excluded.
- Validation with counted rejection reasons: assistant preamble, length drift in both
  directions, near-duplication of the source.
- Runner inherits source_group_id and split from the human document, asserted before write.
- Fixed: the frozen `[mirror_v1 ...]` header comment was being sent to the model.

## Phase 3 - detector, token labels, calibration, evaluation lab

Spec amended to v1.2, resolving the two blockers in section 10:
- token ground truth: build BOTH constructions (`splice` and `edit_diff`), plus a
  mandatory `splice_control` of human-to-human splices labelled entirely human. Without
  the control, a token head can score well by detecting discontinuity alone and no
  metric would reveal it.
- held-out API family pinned to Anthropic `claude-sonnet-5`, budget-capped at 20,000
  documents. For the Claude 4.6 generation and later the dateless id is itself a fixed
  snapshot, so it meets the pinned-revision rule.

Implemented and tested:
- token_labels.py: splice, edit-diff and control construction; spans validated to tile
  each document exactly, with no gaps or overlaps
- alignment.py: character spans to per-token labels, majority-overlap rule, special
  tokens set to the ignore index, round-trip back to spans
- dataset.py: each window labelled by its OWN content rather than inheriting the
  document label
- calibration.py: temperature scaling by golden-section search on NLL, numpy only
- lab.py: one threshold fitted on validation and reused across every regime, per-regime
  FPR measurability check, contamination reporting, headline pulled from named regimes
- encoder.py: FORGE-Base with shared encoder, document head and token head

Bugs found and fixed:
- character-level diffing matched coincidental fragments ("n ", "a", "ing ") between
  unrelated texts, scoring a total rewrite at 0.63 instead of ~1.0 and littering AI
  documents with one-character spans labelled human. Diffing is now word-level with a
  minimum unchanged-run floor.
- the fake generator emitted one unbroken block, so splice construction silently
  produced zero documents. It now paragraphs its output, and `why_splice_failed`
  reports the reason when a splice cannot be built.

Not done, and not claimed: the training fit loop. It cannot be verified without a GPU.

## Phase 4 - hard negative mining and the Failure Atlas

Spec amended to v1.3: mined hard negatives are split at the CLUSTER level, not the
document level. Documents within a failure cluster are near-identical, so a random split
puts near-duplicates on both sides and the held-out score measures memorization while
looking like generalization.

Implemented and tested:
- mining.py: Scorer protocol so mining is testable with a fake scorer, confidence gating
  above the production threshold, false-negative mining as well as false-positive, and a
  ledger so repeat rounds do not re-mine the same failures
- embedding.py: deterministic offline hashing embedder plus a sentence-transformers
  embedder for real runs
- clustering.py: HDBSCAN when installed, seeded numpy k-means++ otherwise, with the
  method recorded so a report cannot present k-means output as density-based
- atlas.py: cluster summaries built from metadata only (no model, so no hallucinated
  failure modes), plus quality_warnings() reporting the atlas's own weaknesses
- selection.py: proportional-across-clusters apportionment with a per-cluster floor, and
  deterministic cluster-level holdout
- run.py: one full turn of the flywheel as a committable JSON artifact

Measured offline on a five-mode synthetic failure set, budget 200:
  proportional selection covered 4 of 5 modes; global top-k covered 1 of 5.

Known limitation, recorded: selection guarantees coverage of clusters, not of true
failure modes. On that same set the smallest mode was absorbed into another cluster and
never selected, and one mode was split across two clusters and over-weighted. The atlas
flags all three conditions rather than hiding them.

`forge mine` refuses to run without a checkpoint rather than producing anything.
