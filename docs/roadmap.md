# Roadmap

Each phase has an exit criterion. A phase is not done because the code exists; it is
done when the criterion is measured and the measurement is committed.

| Phase | Work | Exit criterion |
|---|---|---|
| 0 | Scaffold, Data Spec v1 | `make test` and `make spec-check` pass in CI |
| 1 | FORGE-HUMAN v0.1 | pipeline **done and verified offline**; exit needs 400k train + 50k eval + 5M reserve in Parquet with MANIFEST and zero group leakage, which requires network access to HuggingFace |
| 2 | FORGE-MIRROR v0.1 | engine **done and verified offline**; exit needs 400k real mirrors, pinned revisions, rejection rate reported, length/topic match distributions plotted |
| 3 | Baseline detector | FPR, FNR, AUROC, ECE on R1 committed with the run that produced them |
| 4 | Hard negative loop | first mining pass produces >=10 distinct failure clusters; retrained model's FPR compared to Phase 3 at equal data budget |
| 5 | External evaluation | RAID-extra, MAGE-test, HC3 numbers with a passing contamination check |
| 6 | Adversarial lab | delta-FNR per attack table, ours and RAID's |
| 7 | Distributed + profiling | FSDP vs DeepSpeed throughput on identical config; one profiled bottleneck fixed with before/after traces |
| 8 | Production | API + release gate + monitoring + UI; gate blocks at least one real candidate |

## Blocking questions carried from the spec

1. Token-level ground truth for `ai_assisted` must be decided before Phase 3.
2. The held-out API generator and its budget must be decided before Phase 5.
3. Reserve pool size is a guess; revisit after the first mining pass.
