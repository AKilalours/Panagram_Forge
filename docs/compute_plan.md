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

---

## 6. The under-$20 configuration

### Correction to section 2

Section 2 says Kaggle is disqualified. That is true **at plan scale** (400k mirrors, 7-8B
generators, a 5M reserve pool). It is not true at reduced scale, and stating it flatly was
too strong. At the configuration below most phases fit inside Kaggle's free quota, and the
only thing that genuinely requires paying is the 4-GPU distributed benchmark.

### The four levers, in order of impact

**1. Smaller generator families.** Swap 7-8B for 1.5-4B: Qwen2.5-3B-Instruct,
Llama-3.2-3B-Instruct, Phi-3.5-mini-3.8B, Gemma-2-2B. Throughput roughly triples.

These are still real, distinct generator families, which is what the research question
needs. **Recorded limitation:** smaller models produce more detectable text, so absolute
FPR and FNR will look better than a 7B-to-70B setup would produce. The comparative result
between arms is unaffected; the absolute numbers must not be quoted as if they came from
frontier-scale generators.

**2. Shorter documents.** Filter the human corpus to 150-400 words. Output tokens drop
roughly 2x, and at ~400 tokens a document is exactly one 512-token window.

This is a defensible design choice, not a shortcut: FORGE v1 becomes a single-window
detector, with multi-window aggregation evaluated separately on a small long-document set.
**Recorded limitation:** the windowing and aggregation path is then under-tested, and the
corpus is biased toward short-form web text.

**3. Fewer mirrors.** 60k instead of 400k. 120k training examples is ample for a
DeBERTa-base-sized encoder, and the claim is comparative at equal budget per arm.

**4. Smaller reserve pool.** 500k instead of 5M. **Recorded limitation:** fewer distinct
failure clusters for mining to find, so the Failure Atlas will be thinner and open
question 4 (reserve pool sizing) stays open.

### Cost at that configuration

| Phase | Work | Pod | Hours | Cost |
|---|---|---|---:|---:|
| 2 | 60k mirrors, 18M output tokens, 3B models | 1x RTX 4090 | 1.5 | $1.11 |
| 2 | held-out family, 10k docs for R3 only | 1x RTX 4090 | 0.4 | $0.30 |
| 3 | 4-arm ablation, 360k examples each | 1x A40 | 3.6 | $1.60 |
| 4 | mining, 500k docs x 2 rounds | 1x A40 | 0.2 | $0.08 |
| 5 | RAID-extra subsample, MAGE, HC3 | 1x A40 | 0.1 | $0.03 |
| 6 | paraphrase attacks, 20k docs | 1x RTX 4090 | 0.4 | $0.31 |
| 7 | FSDP vs DeepSpeed, 1/2/4 GPU | 4x A40 | 3.5 | $1.54 |
| | **Compute** | | | **$4.97** |
| | 3x buffer for failed runs | | | $14.91 |
| | 50 GB network volume, one month | | | $3.50 |
| | **All-in** | | | **~$18** |

### Pushing it toward $5

Everything except Phase 7 can run on Kaggle's free 2x T4 at this scale. Phase 2 becomes
roughly 2.5 hours of wall time across two GPUs, well inside a 12-hour session and the
30-hour weekly quota.

**One gotcha that will cost you a week if you hit it blind.** T4 is Turing and has no
bf16, so Kaggle training runs in fp16, and **DeBERTa-v3 in fp16 is known to produce NaN
losses** because of overflow in its disentangled attention. Three options:

- train fp32 on T4, roughly half speed, still feasible at this scale
- use a fp16-safe backbone such as RoBERTa-base for the Kaggle runs
- spend $1.60 on an A40 and train in bf16 as designed

The third is the right answer. Spending $2 to avoid a week of debugging NaN losses is not
a close call.

**Floor: about $2.** One hour on 4x A40 to get the 4-GPU point in the Phase 7 benchmark,
with everything else on Kaggle and the MacBook. Kaggle's 2x T4 can produce a genuine 1-to-2
GPU scaling-efficiency measurement on its own, so even that hour is optional. What you
cannot honestly say without it is "trained across 4 GPUs".

### What this configuration does NOT cost you

- The comparative claim. A/B/C/D at equal data budget per arm is fully intact.
- Every leakage, contamination and measurement-validity guard.
- The multi-GPU distributed story, provided the 4x A40 hour is bought.
- The Failure Atlas loop, at reduced resolution.

Scale up later if the quarter-scale result justifies it. The final table can be regenerated
at plan scale for about $74 once the loop is known to work.
