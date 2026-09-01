# Compute plan

What running the full plan actually costs, and where each phase should run.

Throughput figures are conservative mid-range estimates for bf16 inference and training.
They are estimates, clearly labelled as such, and should be replaced with measurements
from the first real run.

---

## 1. What the plan requires

| Phase | Work | GPU-hours (A100) | Cost @ $1.39/h |
|---|---|---:|---:|
| 1 | Ingestion, 400k train + 50k eval + 5M reserve | **0** (CPU + network) | $0 |
| 2 | 400k mirrors, 4 families, ~200M output tokens | 22 | $31 |
| 3 | 4-run ablation, DeBERTa-v3-base, 2.4M examples each | 15 | $21 |
| 4 | Mining, 3 passes over 5M reserve documents | 5 | $7 |
| 5 | External eval on RAID-extra, MAGE, HC3 | 1 | $2 |
| 6 | Paraphrase and humaniser generation, ~75k documents | 4 | $5 |
| 7 | FSDP vs DeepSpeed matrix, 6 configs across 1/2/4 GPUs | 7 | $10 |
| 8 | Serving, API, UI, monitoring | 0 (CPU) | $0 |
| | **Total** | **~53** | **~$74** |

Budget roughly **$200** in practice. The difference is failed runs, a generation job that
dies at hour eight, an ablation re-run after a config fix, and the reruns that follow the
first time the evaluation lab reports something surprising.

Storage: a 100 GB network volume is about $7/month, which covers the reserve pool,
mirrors and checkpoints.

---

## 2. Platform verdict

| | Kaggle | Colab Pro+ | RunPod |
|---|---|---|---|
| Cost | free | ~$50/mo | pay per hour |
| GPUs per session | 2x T4 (16 GB) | **1** | 1 to 8 |
| bf16 | **no** (T4 is Turing) | yes on L4/A100 | yes |
| Session limit | 12 h | disconnects | none |
| Weekly quota | 30 h | compute units | none |
| Persistent storage | ~20 GB | Drive | network volumes |
| Docker | no | no | yes |
| **Runs Phase 2 at scale** | no | painful | yes |
| **Runs Phase 7 at all** | partially | **no** | yes |

**Colab is disqualified structurally, not on price.** One GPU per session means Phase 7
cannot run, and Phase 7 is the part that speaks directly to the JD's multi-GPU and
distributed-training requirement. No amount of Pro+ fixes that.

**Kaggle is disqualified on throughput and storage.** T4 has no bf16, and is roughly 3 to
4 times slower than a 4090 on this workload. Phase 2 alone would take over 150 T4-hours,
which is five weeks of the free quota, and the 5M-document reserve pool does not fit in
the persistent working directory.

**RunPod is the only option that runs the plan as written.** Multi-GPU pods, persistent
network volumes, and Docker support, so `infra/docker/Dockerfile.train` is used as-is
rather than rewritten as notebook cells.

---

## 3. Where each phase should actually run

### On the MacBook (free)
- **Phase 1 ingestion.** Needs network, disk and CPU hours, no GPU. Streaming FineWeb
  sample-10BT and filtering to the reserve pool is an overnight job.
- **Phase 8 serving, API, UI.** CPU inference is fine for a demo.
- The whole test suite.

### On Kaggle (free, one specific job)
- **Phase 7 correctness debugging on 2x T4.** Multi-GPU code has failure modes that only
  appear at world_size > 1: wrapping policies, checkpoint sharding, sampler seeding,
  collective deadlocks. Find those for free before paying for the benchmark.
- Note T4 has no bf16, so debug in fp16 and treat any throughput number from Kaggle as
  meaningless. This is a correctness run, not a measurement.

### On RunPod (paid)

| Phase | Pod | Rate | Why |
|---|---|---|---|
| 2 generation | 1x A100 80GB | $1.39/h | large KV cache raises vLLM throughput; finishes in half the wall time of a 4090 |
| 2 budget option | 1x RTX 4090 | $0.74/h | 7-8B fits in 24 GB bf16; similar cost per token, twice the wall time |
| 3 training | 1x A100 80GB | $1.39/h | |
| 4 mining | 1x A40 48GB | $0.44/h | inference only, no need for A100 |
| 7 distributed | **4x A40 48GB** | $1.76/h total | Ampere, so bf16 works. 4x A100 would be $5.56/h for a benchmark that does not need the memory |

The A40 at $0.44 is the value pick for anything that is not latency-bound. It supports
bf16, which the T4 does not, and four of them cost a third of four A100s.

**Use spot or community instances where offered.** They are cheaper and can be preempted,
which is exactly the case `training/train.py` checkpoint resume was written for. A run
that cannot resume turns every preemption into a lost day; this one does not.

---

## 4. The most important cost lever

**Start at one quarter scale.**

| | Plan scale | Quarter scale |
|---|---|---|
| Mirrors | 400k | 100k |
| Reserve pool | 5M | 1.25M |
| GPU-hours | ~53 | ~21 |
| Cost | ~$74 | ~$29 |

The research claim is **comparative**: mirrors versus random synthetic, hard negatives
versus none, at an equal data budget in each arm. Absolute corpus size does not change
what the ablation proves. A clean result at 100k mirrors is a result; a broken run at
400k is twenty hours of GPU time and no result.

Run the loop end to end at quarter scale first, confirm the numbers move in the direction
the design predicts, then scale up for the final table. The first full-scale run will
have bugs, and the cheapest place to find them is in a run that costs $29.

---

## 5. Order of operations

1. Ingestion on the MacBook, quarter scale, overnight. No cost.
2. RunPod A100, one hour, generate 5k mirrors. Verify the mirror validation rejection
   rate and the length-match distribution before committing to a long job.
3. RunPod A100, generate 100k mirrors. Roughly 6 hours.
4. RunPod A100, train the baseline. Roughly 1 hour at quarter scale. **First real
   numbers in `docs/evaluation.md`.**
5. Kaggle 2x T4, debug FSDP and DeepSpeed for free.
6. RunPod 4x A40, run the Phase 7 benchmark matrix.
7. Full 4-run ablation, mining rounds, external evaluation.
8. Scale to plan size only if the quarter-scale result justifies it.

Step 4 is the one that matters. It is the first point at which this project has a
measured result rather than an implementation.
