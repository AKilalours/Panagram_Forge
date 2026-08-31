# JD coverage matrix

Where each required capability lives in FORGE, which phase implements it for real, and
what would make the claim defensible in an interview.

The rule this table follows: a row is only marked **implemented** when there is code
that runs and a measurement committed alongside it. Everything else says *planned*
with the phase that owns it. An interviewer who opens the repo should find the state
exactly as described here.

| # | Requirement | Where in FORGE | Phase | State |
|---|---|---|---|---|
| 1 | Python and modern ML frameworks | whole `src/forge` tree, PyTorch + HF Transformers | 3 | scaffolded |
| 2 | Transformers and LLM fundamentals | `modeling/encoder.py` (encoder + dual head), `modeling/windowing.py` (overlapping windows), `generation/` (decoding grid, sampling) | 3 | windowing implemented and tested; model planned |
| 3 | Research and engineering boundaries | `docs/data_spec_v1.md` (frozen contract), `evaluation/release_gate.py`, `configs/eval/regimes.yaml` | 0 | **implemented** |
| 4 | NVIDIA GPU programming and CUDA | `infra/docker/Dockerfile.train` (CUDA 12.4 base), `kernels/cuda/` | 7 | planned |
| 5 | Distributed training (DeepSpeed, FSDP, Ray) | `training/distributed.py`, `orchestration/ray/` | 7 | planned |
| 6 | Inference frameworks (vLLM) | `generation/generators/base.py` (vLLM for batch generation), `inference/server.py` (deliberately *not* vLLM) | 2 and 8 | planned |
| 7 | Large-scale data processing (Spark, Beam) | `orchestration/spark/`, `cleaning/pipeline.py` | 1 then 7 | planned |
| 8 | Orchestration (Airflow) | `orchestration/airflow/dags/` | 8 | planned |
| 9 | MLOps and experiment tracking | `registry/model_registry.py`, W&B config in every training YAML, `MANIFEST.json` dataset versioning | 3 | registry contract implemented |
| 10 | DevOps tools | `.github/workflows/ci.yml`, `Makefile`, `infra/docker/`, `pyproject.toml`, ruff/mypy/pytest | 0 | **implemented** |
| 11 | Cloud infrastructure (AWS/GCP) | `infra/terraform/`, `infra/aws/`, `infra/gcp/`, S3/MinIO storage layout in spec section 8 | 7 and 8 | planned |

---

## The honest version of each claim

**CUDA (#4).** The defensible target is one thing done properly, not a fake kernel.
Two candidates, both real work: a fused windowing-plus-tokenization preprocessing
kernel, or a profiled attention path where a measured bottleneck is fixed and the
before/after trace is committed. Phase 7 picks one after profiling says which is
actually the bottleneck. Writing a CUDA kernel for a stage that is 3 percent of step
time is resume decoration, not engineering.

**vLLM (#6).** vLLM belongs on the generation side, where FORGE decodes hundreds of
thousands of documents from open-weight models and continuous batching genuinely
helps. It does not belong on the detector's serving path: FORGE-Base is a
bidirectional encoder with a fixed 512-token window and no KV cache, so vLLM's core
optimizations do not apply. Being able to explain *why not* is worth more than a
misapplied dependency.

**Spark / Beam (#7).** Phase 1 runs on Polars and PyArrow because 400k documents fit
on one machine and Spark would add operational cost for no throughput. Spark earns its
place in Phase 7, at the 5-million-document reserve pool, where the mining scan is
genuinely embarrassingly parallel over shards. `orchestration/spark/` holds the job
that does that scan. Forcing Spark into v1 would be a worse engineering answer, and an
interviewer who probes will find that out.

**Airflow (#8).** The recurring pipeline is the flywheel itself: scan reserve pool,
cluster failures, generate targeted mirrors, retrain, evaluate, gate. That is a real
DAG with real dependencies and retry semantics, so it is worth orchestrating. A DAG
that runs one script on a schedule is not.

**AWS/GCP (#11).** Local development uses MinIO with the identical S3 layout from spec
section 8, so the cloud move is a config change rather than a rewrite. Terraform
defines the training bucket, the GPU instance profile and the registry. The measured
claim to aim for is a cost-per-training-run number, which is far more convincing than
a list of service names.

**Experiment tracking (#9).** The claim worth making is not "used W&B". It is that
every run is reproducible from `dataset_version` plus `code_commit` plus config, and
that `registry/model_registry.py` refuses an entry missing any of them. That is
already enforced by a test.
