# Roadmap

Each phase has an exit criterion. A phase is not done because the code exists; it is
done when the criterion is measured and the measurement is committed.

| Phase | Work | Exit criterion |
|---|---|---|
| 0 | Scaffold, Data Spec v1 | `make test` and `make spec-check` pass in CI |
| 1 | FORGE-HUMAN v0.1 | pipeline **done and verified offline**; exit needs 400k train + 50k eval + 5M reserve in Parquet with MANIFEST and zero group leakage, which requires network access to HuggingFace |
| 2 | FORGE-MIRROR v0.1 | engine **done and verified offline**; exit needs 400k real mirrors, pinned revisions, rejection rate reported, length/topic match distributions plotted |
| 3 | Baseline detector | components **done and tested**; exit needs FPR, FNR, AUROC, ECE on R1 committed with the run that produced them, which requires a GPU |
| 4 | Hard negative loop | components **done and tested**; exit needs a first mining pass producing >=10 distinct failure clusters with acceptable silhouette, and a retrained model's FPR compared to Phase 3 at equal data budget, on both trained and HELD-OUT failure clusters |
| 5 | External evaluation | loaders **done and tested against schema fixtures**; exit needs RAID-extra, MAGE-test and HC3 numbers with a passing contamination check, which requires network access and a trained model |
| 6 | Adversarial lab | attacks and runner **done and tested**; exit needs the delta-FNR table in both preprocessing conditions for a real model, plus RAID's 12 attacks, plus a real paraphrase model for the 4 model-based attacks |
| 7 | Distributed + profiling | scaling math and configs **done and tested**; exit needs FSDP vs DeepSpeed throughput at an ASSERTED-identical global batch, plus one profiled bottleneck fixed with before/after traces |
| 8 | Production | components **done and tested**; exit needs the gate to block at least one real candidate, and a canary promotion on real traffic |

## Blocking questions

Resolved in spec v1.2:
1. ~~Token-level ground truth for `ai_assisted`~~ - build both constructions plus a
   human-to-human control.
2. ~~The held-out API generator and its budget~~ - Anthropic `claude-sonnet-5`, capped
   at 20,000 documents, evaluation only.

Still open, neither blocking a phase:
3. Does prediction smoothing improve boundary F1? A Phase 3 ablation, not a design choice.
4. Reserve pool size is a guess; revisit after the first mining pass.
5. Light-touch AI editing (a few words per sentence in a long document) falls below the
   0.05 edit-fraction floor and cannot be labelled by post-hoc diffing. Needs provenance
   from the editor itself.
6. Selection covers clusters, not true failure modes. A merged cluster starves the
   smaller mode inside it. Mitigated by atlas quality warnings; a real fix needs better
   separation (semantic embeddings plus HDBSCAN) and should be re-measured once real
   failures exist.
