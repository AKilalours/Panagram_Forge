# FORGE

**Failure-Driven Synthetic Data Generation for Robust AI-Content Detection**

## Research question

> Can failure-driven synthetic data generation produce a detector that generalizes
> better to unseen LLMs and adversarial transformations than conventional random
> synthetic-data training, while maintaining an extremely low human false-positive rate?

## Status

Tier 1 is complete: both arms trained, evaluated in distribution and against three
held-out-generator benchmarks, with every number committed alongside the run that produced
it. Read `docs/writeup.md` first, then `docs/evaluation.md` for the full tables.

| Phase | What it is | State |
|---|---|---|
| 0 | Repo scaffold, Data Spec v1 frozen | done |
| 1 | FORGE-HUMAN ingestion (FineWeb, FineWeb-Edu, public domain) | **done**, 60,000 documents, `dataset_version` v0.1-min |
| 2 | Synthetic mirror engine, FORGE-MIRROR v0.1 | **done**, 30,000 AI documents per arm from four families at 1.7B to 3.8B |
| 3 | Baseline detector (encoder + document head + token head) | **done**, arms A and B trained, DeBERTa-v3-base, one RTX 4090 |
| 4 | Hard negative mining loop | mining, atlas, clustering and selection built and tested; **the CLI entrypoint is not yet wired to them, and arm C has not run** |
| 5 | External evaluation (RAID, MAGE, HC3) | **done**, six cells, paired bootstrap and paired McNemar, score arrays committed |
| 6 | Adversarial laboratory | 8 offline attacks + lab runner built and tested; **entrypoint not wired**; 4 model-based attacks refuse rather than fake |
| 7 | Distributed training (FSDP / DeepSpeed / Ray), profiling | scaling math, config generation and the batch-invariant benchmark matrix built and tested; **needs 2 GPUs to measure, no scaling claim is made** |
| 8 | Production serving, release gate, monitoring, UI | decision policy with abstention, batching, drift, feedback state machine, registry promotion and UI built and tested |

Every number in this repo is either measured and committed with the run that
produced it, or absent. There are no placeholder metrics in results tables.

## The Tier 1 result, in three lines

- In distribution the two arms cannot be separated: 0.719% against 0.430% FNR at an
  identical 0.1% false-positive budget, **p = 0.217**, and AUROC 0.99997 on both.
- Out of distribution the AUROC story does not hold up. On RAID the arms are tied and the
  sign reverses in 72.7% of bootstrap resamples. But at a **matched** 0.1% FPR budget the
  mirror arm catches **122** AI documents the control misses on RAID while losing nine the
  other way, with no AUROC difference at all. Mirroring moved the low-false-positive tail
  without moving the ranking.
- **Neither arm is deployable off distribution.** Both miss 63% to 96% of AI text at a
  usable threshold, and ECE goes from 0.004 in distribution to 0.18-0.44 outside it.

**What is not yet tested is the thesis in the title.** Arm B is matched mirroring, not
failure-driven selection. Arm C, hard negatives mined from arm B's own failures, is the arm
that tests failure-driven generation, and it has not run.

## The flywheel

```
production -> failures -> failure atlas -> targeted mirror generation
   ^                                                    |
   |                                                    v
release gate <- evaluation lab <- training <- targeted dataset
```

## Quick start

```bash
make setup          # venv + editable install
make test           # 55 tests
make spec-check     # configs must match the frozen Data Spec v1

# offline smoke runs, no network or GPU needed
make smoke          # Phase 1: ingestion
make smoke-mirror   # Phase 2: mirror engine, fake generator

# the real thing (needs network and the data extra)
make install-data
make ingest
```

`make smoke` generates a synthetic fixture corpus and runs ingestion end to end.
The fixture is a pipeline test, never training data.

## Key design decisions

1. **Text first.** Image and video detection are out of scope for v1. Breadth here
   would produce a shallow project.
2. **Group-aware splits.** A human document and every synthetic mirror derived from
   it share a `source_group_id` and always land in the same split. Random splitting
   after mirroring leaks semantics and inflates every metric.
3. **FPR is the primary metric.** Accusing a human is the expensive error. Accuracy
   and AUROC are reported but never used as the release criterion.
4. **RAID and MAGE are evaluation only.** They never enter the training set. Any
   contamination invalidates the headline result.
5. **Provenance on every document.** Source, license, acquisition date, processing
   version and content hash are stored per record. Where redistribution is
   restricted, only IDs and metadata are committed, never text.
6. **No training on raw user feedback.** Thumbs-down goes to a verification queue,
   not to the training set.

## Layout

See `docs/architecture.md` for the full system diagram, `docs/data_spec_v1.md` for
the frozen data contract, and `docs/jd_coverage.md` for where each infrastructure
capability lives and which phase implements it.

## Licensing

Code: Apache-2.0. Data: per-source, tracked in `docs/data_card.md`. This repository
does not redistribute any corpus whose license forbids it.
