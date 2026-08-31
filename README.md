# FORGE

**Failure-Driven Synthetic Data Generation for Robust AI-Content Detection**

## Research question

> Can failure-driven synthetic data generation produce a detector that generalizes
> better to unseen LLMs and adversarial transformations than conventional random
> synthetic-data training, while maintaining an extremely low human false-positive rate?

## Status

| Phase | What it is | State |
|---|---|---|
| 0 | Repo scaffold, Data Spec v1 frozen | done |
| 1 | FORGE-HUMAN ingestion (FineWeb, FineWeb-Edu, public domain) | pipeline built and verified offline; real ingest not yet run |
| 2 | Synthetic mirror engine, FORGE-MIRROR v0.1 | engine built and verified offline; real generation not yet run |
| 3 | Baseline detector (encoder + document head + token head) | not started |
| 4 | Hard negative mining loop | not started |
| 5 | External evaluation (RAID, MAGE, HC3) | not started |
| 6 | Adversarial laboratory | not started |
| 7 | Distributed training (FSDP / DeepSpeed / Ray), profiling | not started |
| 8 | Production serving, release gate, monitoring, UI | not started |

Every number in this repo is either measured and committed with the run that
produced it, or absent. There are no placeholder metrics in results tables.

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
