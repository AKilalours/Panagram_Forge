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
