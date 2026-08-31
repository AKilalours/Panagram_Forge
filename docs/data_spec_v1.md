# FORGE Data Specification v1 (FROZEN)

Status: **FROZEN** as of 2026-08-31.
Any change to this document requires a version bump to v1.1 and a re-run of
`make spec-check`. Configs under `configs/` are validated against this file in CI.

The purpose of freezing this is simple: the ingestion pipeline, the mirror engine,
the split logic and the evaluation lab all depend on these decisions. If they move
after code exists, every dataset version built before the change becomes
incomparable and the ablation table in `docs/evaluation.md` becomes meaningless.

---

## 1. Dataset inventory

Every source is classified as **TRAIN**, **EVAL-ONLY**, or **RESERVE**.
A source may never change class. EVAL-ONLY sources entering training is the single
failure that would invalidate the entire project's headline result.

| ID | Source | HF repo / origin | License | Class | Role |
|----|--------|------------------|---------|-------|------|
| `fw` | FineWeb | `HuggingFaceFW/fineweb`, config `sample-10BT` | ODC-By 1.0 (Common Crawl ToU also applies) | TRAIN + RESERVE | bulk human web text |
| `fwe` | FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, config `sample-10BT` | ODC-By 1.0 | TRAIN + RESERVE | high quality explanatory/academic-like human prose |
| `gut` | Project Gutenberg | gutenberg.org, PG-only texts | Public domain (US) | TRAIN | long-form literary human text |
| `gov` | US federal documents | govinfo.org bulk data | Public domain (US Gov work) | TRAIN | formal/bureaucratic human register |
| `hc3` | HC3 | `Hello-SimpleAI/HC3` | CC-BY-SA 4.0 | EVAL-ONLY | human vs ChatGPT QA, legacy generator check |
| `raid` | RAID | `liamdugan/raid` | MIT | EVAL-ONLY | generator / domain / decoding / adversarial shift |
| `mage` | MAGE | `yaful/MAGE` | Apache-2.0 | EVAL-ONLY | in-the-wild OOD, 322 distinct sources |

### 1.1 Verified facts (checked against live dataset cards, 2026-08-31)

**FineWeb** (`HuggingFaceFW/fineweb`)
- Configs: `default` (114 CC dumps, ~18.5T gpt2 tokens, ~50.4 TB), `sample-350BT`,
  `sample-100BT`, `sample-10BT` (~27.6 GB), plus per-dump `CC-MAIN-YYYY-WW`.
- Fields: `text`, `id`, `dump`, `url`, `date`, `file_path`, `language`,
  `language_score`, `token_count`.
- **Decision: v1 uses `sample-10BT`, not `sample-100BT`.** The plan called for
  100BT. 277 GB is not downloadable or storable on the dev machine, and the
  hard-negative mining pool only needs to be large relative to the training set,
  not absolutely large. 10BT gives roughly 15 million documents, which is far more
  than the 5 to 20 million target. If mining saturates, escalate to `sample-100BT`
  on cloud storage in Phase 7, not before.

**FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`)
- Same field set plus `score` (float educational quality, ~2.52 to 5.06) and
  `int_score` (binned, 3 to 5). 1.3T tokens, 1.53B rows total.
- v1 uses config `sample-10BT`, then filters to `int_score >= 4`.

**HC3** (`Hello-SimpleAI/HC3`)
- Configs: `all`, `finance`, `medicine`, `open_qa`, `reddit_eli5`, `wiki_csai`.
- Fields: `id`, `question`, `human_answers` (list), `chatgpt_answers` (list), `source`.
- Note the answers are **lists**, so one row expands to several documents. Row
  counts are not document counts.

**RAID** (`liamdugan/raid`, MIT)
- 11 models: ChatGPT, GPT-4, GPT-3, GPT-2, Llama-Chat, Mistral, Mistral-Chat,
  MPT, MPT-Chat, Cohere, Cohere-Chat.
- 11 domains: abstracts, books, code, Czech, German, news, poetry, recipes,
  Reddit, reviews, Wikipedia.
- 4 decoding settings: greedy and sampling, each with and without repetition
  penalty 1.2.
- 12 adversarial attacks including homoglyph, zero-width space, whitespace,
  article deletion, insert paragraphs, perplexity misspelling, upper/lower case,
  synonym, paraphrase, alternative spelling, number.
- Fields: `id`, `adv_source_id`, `source_id`, `model`, `decoding`,
  `repetition_penalty`, `attack`, `domain`, `title`, `prompt`, `generation`.
- Splits: `RAID-train` (labeled), `RAID-extra` (labeled), `RAID-test`
  (**labels held out**, leaderboard submission required).
- **Decision:** FORGE evaluates on `RAID-extra` for self-reported numbers, because
  it is labeled and disjoint from RAID-train. `RAID-train` is used only to sanity
  check the harness. Neither ever enters FORGE training. A leaderboard submission
  on `RAID-test` is a Phase 5 stretch goal.
- **Decision:** the `code`, `Czech` and `German` domains are excluded from the
  headline number and reported separately. FORGE v1 is an English natural-language
  detector; scoring it on code and non-English inflates or deflates results for
  reasons unrelated to the research question.

**MAGE** (`yaful/MAGE`, Apache-2.0)
- 436,606 rows. Fields: `text`, `label` (int 0/1), `src` (322 classes).
- Splits: train 319,000 / validation 56,800 / test 60,700.
- **Landmine:** the HF card states `1` = machine-generated, but the original
  DeepfakeTextDetect release used the opposite convention in some artifacts.
  `src/forge/evaluation/ood.py` MUST assert the polarity empirically at load time
  (a known-human source such as a human `src` prefix must score as human) and fail
  loudly rather than silently reporting an inverted AUROC. This is recorded as
  test `tests/unit/test_mage_polarity.py`.

---

## 2. Target volumes for v1

These are targets, not commitments. Under-delivering is fine and gets recorded.
Over-delivering without a reason is scope creep.

| Pool | Documents | Purpose |
|------|-----------|---------|
| `FORGE-HUMAN-train` | 400,000 | human side of the training set |
| `FORGE-HUMAN-reserve` | 5,000,000 | hard negative mining pool, never trained on directly |
| `FORGE-HUMAN-eval` | 50,000 | held out human documents for FPR measurement |
| `FORGE-MIRROR-v0.1` | 400,000 | one AI mirror per training human doc, generator round-robin |
| `FORGE-HARDNEG-v0.1` | 50,000 to 100,000 | mined FP humans plus their targeted mirrors |
| `FORGE-ADV-v0.1` | 100,000 | controlled transformations of held-in AI documents |

Composition of `FORGE-HUMAN-train` (400k):
`fw` 45%, `fwe` 30%, `gut` 15%, `gov` 10%.
The Gutenberg and govinfo slices exist so the model sees registers that Common
Crawl underrepresents (long literary prose, formal legalese). Both are exactly the
registers that produce human false positives in production detectors.

---

## 3. Record schemas

Canonical Pydantic models live in `src/forge/common/schemas.py`. This section is
the contract; the code must match it and `tests/unit/test_schemas.py` enforces it.

### 3.1 HumanDocument

```json
{
  "doc_id": "fw_000000018392",
  "source_group_id": "grp_fw_000000018392",
  "text": "...",
  "source": "fineweb",
  "source_config": "sample-10BT",
  "source_record_id": "<upstream uuid>",
  "license": "ODC-By-1.0",
  "domain": "web",
  "register": "informational",          // python attr is `text_register` (alias), see schemas.py
  "language": "en",
  "language_score": 0.97,
  "date": "2024-03-11",
  "acquired_at": "2026-09-02T14:00:00Z",
  "processing_version": "clean_v1",
  "content_sha256": "...",
  "token_count": 812,
  "quality": { "edu_score": null, "length_ok": true, "pii_flags": [] },
  "split": "train",
  "redistributable": false
}
```

`redistributable` is the flag that decides what gets committed. When `false`, the
`text` field is written only to local Parquet under `data/` (which is gitignored)
and the committed artifact carries ID plus metadata plus hash only. See section 8.

### 3.2 SyntheticDocument

```json
{
  "sample_id": "forge_00018392",
  "source_human_id": "fw_000000018392",
  "source_group_id": "grp_fw_000000018392",
  "label": "ai",
  "text": "...",
  "generator": {
    "provider": "open_source",
    "family": "qwen",
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "<git sha of the HF repo>",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 1024,
    "seed": 42
  },
  "mirror": {
    "prompt_version": "mirror_v1",
    "target_tokens": 812,
    "topic_match": true,
    "length_match": true,
    "style_match": true,
    "attributes": { "genre": "explainer", "register": "informational",          // python attr is `text_register` (alias), see schemas.py "structure": "prose" }
  },
  "domain": "web",
  "language": "en",
  "transformations": [],
  "generated_at": "2026-09-05T09:11:00Z",
  "license": "synthetic",
  "split": "train"
}
```

`generator.revision` is mandatory. "Qwen 7B" is not a reproducible statement;
a repo revision hash is. Model versions are frozen in `configs/generation/generators.yaml`.

### 3.3 TokenLabelSpan

Token-level supervision uses character spans, not token indices, because token
indices are tokenizer-dependent and the tokenizer will change.

```json
{ "start_char": 0, "end_char": 1840, "label": "ai_generated" }
```

Label set (exactly three, ordered): `human`, `ai_assisted`, `ai_generated`.

### 3.4 FailureRecord

```json
{
  "sample_id": "fw_000000018392",
  "true_label": "human",
  "prediction": "ai",
  "confidence": 0.97,
  "domain": "academic",
  "source": "fineweb",
  "register": "informational",          // python attr is `text_register` (alias), see schemas.py
  "embedding_id": "emb_00412",
  "cluster": 17,
  "model_version": "forge-0.3",
  "failure_type": "human_false_positive",
  "discovered_at": "2026-10-01T00:00:00Z",
  "discovered_by": "mining_run_004"
}
```

---

## 4. Split rules (the most important section)

### 4.1 Grouping key

`source_group_id` is assigned **once, at human ingestion time**, and inherited by
every synthetic mirror, every hard negative mirror and every adversarial variant
derived from that human document.

Splitting is performed on `source_group_id`, never on rows.

Concretely: if human doc A produced mirrors A1, A2, A3 and adversarial variants
A1-para, A1-homoglyph, then all seven records carry `grp_A` and all seven land in
whichever split `grp_A` was assigned. A random row split would place A in train and
A2 in test, and the model would score near-perfectly on semantically identical
content it had already memorized.

### 4.2 Assignment

Deterministic and reproducible, not random at runtime:

```
split = bucket(sha256(source_group_id + SPLIT_SALT))
SPLIT_SALT = "forge-v1"
train 0.80 / val 0.10 / test 0.10
```

Because the salt and hash are fixed, re-running ingestion on a superset of
documents keeps every previously assigned document in its original split. This
means dataset v0.2 is directly comparable to v0.1.

`tests/unit/test_splits.py` asserts zero `source_group_id` overlap across splits
and is a required CI check.

### 4.3 The five evaluation regimes

| Regime | Train on | Test on | What it measures |
|---|---|---|---|
| R1 IID | all held-in | held-out split, same distribution | basic competence |
| R2 unseen domain | web, news, reviews, books | academic, poetry, recipes | domain shift |
| R3 unseen generator | Llama, Qwen, Mistral | Gemma, DeepSeek, Command | generator shift |
| R4 temporal | older model generations | newer model generations | the actual production problem |
| R5 adversarial | clean AI | humanized / paraphrased / perturbed | evasion robustness |

Plus three external regimes with zero training contamination: RAID-extra,
MAGE-test, HC3.

R4 is the hardest to construct honestly, because "older" and "newer" must refer to
model release date, not to when FORGE generated the text.
`configs/generation/generators.yaml` records a `released` date per model and R4 is
built from that field.

---

## 5. Generator roster (frozen for v0.1)

Six open-weight families plus one API family. Held-in versus held-out is fixed here
so that the unseen-generator claim is not chosen after seeing results.

| Family | Model id | Released | Role |
|---|---|---|---|
| llama | `meta-llama/Llama-3.1-8B-Instruct` | 2024-07 | held-in |
| qwen | `Qwen/Qwen2.5-7B-Instruct` | 2024-09 | held-in |
| mistral | `mistralai/Mistral-7B-Instruct-v0.3` | 2024-05 | held-in |
| gemma | `google/gemma-2-9b-it` | 2024-06 | **held-out (R3)** |
| deepseek | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 2024-05 | **held-out (R3)** |
| phi | `microsoft/Phi-3.5-mini-instruct` | 2024-08 | held-in |
| api | one frontier API model, recorded at run time | varies | **held-out (R3 + R4)** |

Rules:
1. Held-out families never appear in any training split, including hard negative
   mirrors and adversarial variants.
2. Every generation records the exact repo revision. A model updated upstream is a
   different generator and gets a new row.
3. Decoding is sampled per document from a fixed grid so the detector cannot latch
   onto one temperature: `temperature` in {0.5, 0.7, 1.0}, `top_p` in {0.9, 0.95},
   plus a greedy setting. The chosen values are stored per record.

---

## 6. Mirror prompt contract

`mirror_v1` is frozen. New prompt wording means `mirror_v2` and a new dataset version.

Pipeline per human document:
1. **Extract attributes** with a small local model or deterministic heuristics:
   topic, genre, register, target token count, structural shape (prose / listy /
   sectioned / dialogue), reading difficulty, and 3 to 6 key factual anchors.
2. **Render** those attributes into the mirror prompt template at
   `src/forge/generation/prompts/mirror_v1.txt`.
3. **Generate** with the assigned generator and decoding setting.
4. **Validate**: reject and retry (max 2 retries) if the output length is outside
   0.6x to 1.6x of the target, if it opens with an assistant preamble
   ("Sure, here is..."), or if it is near-duplicate of the human source above a
   MinHash Jaccard of 0.5.

Attribute extraction never copies sentences from the human document. If it did, the
mirror would share surface text with its human counterpart and the classifier would
learn a copy-detection shortcut rather than a generation-process signal.

The goal, stated precisely: match P(topic, genre, length, structure) between the
human and AI sides so that the only remaining systematic difference is the
generation process itself.

---

## 7. Processing pipeline contract

Order is fixed. Each stage writes a new layer and never mutates the previous one.

```
raw    -> exactly as downloaded, immutable, hash recorded
bronze -> schema-validated, one row per document, Parquet
silver -> normalized, filtered, deduplicated, PII-scrubbed, split assigned
gold   -> training-ready, joined human + synthetic + hard negative + adversarial
```

Stage order in `src/forge/cleaning`:

1. schema validation
2. language ID (keep `en`, `language_score >= 0.85`)
3. Unicode NFKC normalization plus ftfy mojibake repair
4. HTML and markup removal
5. length filter (min 200 chars, min 50 tokens, max 20,000 tokens)
6. quality scoring (symbol ratio, repeated-line ratio, mean word length)
7. PII detection and redaction (email, phone, SSN-like, IBAN-like)
8. exact dedup on `content_sha256`
9. near-duplicate dedup, MinHash LSH, 128 permutations, Jaccard threshold 0.8
10. domain and register classification
11. `source_group_id` assignment and split bucketing
12. Parquet write, partitioned by `source` and `split`

Important ordering note: **dedup runs before split assignment.** Deduplicating after
splitting leaves near-duplicate pairs straddling train and test, which is the same
leak as section 4 in a different disguise.

---

## 8. Storage layout and redistribution policy

```
s3://forge-data/          (MinIO locally, S3 in cloud)
  raw/<source>/<acquired_date>/
  bronze/<source>/
  silver/<source>/split=<split>/
  gold/<dataset_version>/
  eval/<benchmark>/
  reserve/<source>/
```

Local `data/` mirrors this layout and is fully gitignored.

**Redistribution rule.** FORGE does not republish any corpus text. What is committed
to the repository, and what is publishable alongside the paper or writeup, is:

- `doc_id`, `source`, `source_record_id`, `license`, `content_sha256`
- all metadata and labels
- the exact code and configs that regenerate the dataset from the upstream sources

Synthetic text FORGE generated is redistributable, but only where the generator's
license permits output redistribution. `configs/generation/generators.yaml` carries
an `output_redistributable` boolean per family and the packaging step honors it.

FineWeb and FineWeb-Edu are ODC-By 1.0 but the underlying web content carries its
own rights and Common Crawl's terms of use still apply. Treat them as
non-redistributable in FORGE regardless of the ODC-By label.

---

## 9. Dataset versioning

`FORGE-<POOL>-v<major>.<minor>`, recorded in `data/gold/<version>/MANIFEST.json`:

```json
{
  "dataset_version": "v0.1",
  "created_at": "...",
  "code_commit": "<git sha>",
  "spec_version": "data_spec_v1",
  "sources": [{ "id": "fw", "config": "sample-10BT", "n_docs": 180000, "sha256_of_ids": "..." }],
  "counts": { "human": 400000, "ai": 400000, "hard_negative": 0, "adversarial": 0 },
  "split_salt": "forge-v1"
}
```

A training run that cannot name its `dataset_version` and `code_commit` is not a
result, it is an anecdote.

---

## 10. Open questions deliberately left unresolved

These are recorded rather than guessed. Each blocks a specific later phase, not this one.

1. **Token-level ground truth for `ai_assisted`.** Mirrors give clean document-level
   labels. Genuine mixed human/AI documents with trustworthy character-level
   boundaries need a construction procedure (splice at paragraph boundaries?
   LLM-edit a human document and diff?). Blocks Phase 3's token head. Decide before
   Phase 3 starts.
2. **Which frontier API model** to use for the held-out `api` family, and the budget
   for it. Blocks R3 and R4 completeness.
3. **Whether smoothing helps.** Section 16 of the plan assumes token-prediction
   smoothing improves boundary F1. This is an experiment, not a design decision.
   Blocks nothing; it is a Phase 3 ablation.
4. **Reserve pool size for mining.** 5M is a guess. The right number is whatever
   makes the FP rate at the mining threshold yield enough distinct clusters. Revisit
   after the first mining pass.

---

## 11. Spec changelog

### v1.1 (2026-08-31) - stage reorder in section 7

**Change.** The length filter moves from position 5 to position 2, ahead of language
identification. New order: schema validation, unicode normalization, markup removal,
length filter, language id, quality scoring, PII, exact dedup, near-duplicate dedup,
domain classification, split assignment, Parquet write.

**Why.** Found during the Phase 1 offline verification run. A fixture document reading
`"Too short."` was rejected with reason `language`, because language detection ran
first and could not identify a two-word string. The rejection was technically true and
completely unhelpful: the ingestion report attributed a length failure to a language
failure. At 5M documents that misattribution would hide the real shape of what the
pipeline is discarding.

Two supporting reasons. Running language detection on documents that are about to be
dropped for length is wasted compute, and length is the only gate in the pipeline that
is unambiguous and tokenizer-free, so it is the correct cheapest-first filter.

**What did not change.** Dedup still runs before split assignment, and markup removal
still runs before the length filter. Both of those orderings are correctness
properties, not performance choices.

**Regression guard.** `tests/unit/test_pipeline.py::test_short_document_is_rejected_for_length_not_language`.

### v1.2 (2026-08-31) - section 10 open questions 1 and 2 resolved

**Question 1, token-level ground truth for `ai_assisted`. Resolved: build both sources.**

Two constructions, recorded per record in a `construction` field so their contributions
can be separated in analysis:

- `splice` - a mixed document is built by interleaving paragraphs from a human document
  with paragraphs from that same document's own mirror. Boundaries are exact and free,
  it runs offline, and it can be built today. Its weakness is that the seams are
  abruptly discontinuous, so a token head could learn "topic or style shift" rather than
  "authorship shift".
- `edit_diff` - a human document is partially rewritten by a model and the character
  diff gives the `ai_assisted` spans. This matches how people actually use AI, which is
  the case the product cares about. It costs GPU or API time per document.

**Mandatory control.** A third construction, `splice_control`, interleaves paragraphs
from two different HUMAN documents and labels every span `human`. Without it, the token
head can reach high accuracy by detecting discontinuity alone, and nothing in the metrics
would reveal that. The control set is not optional and its size tracks the splice set.

Splices and edit-diffs inherit `source_group_id` and `split` from their human source,
exactly like mirrors. A splice built from two different human documents inherits the
group of the first and is only constructed from documents already in the same split.

**Question 2, the held-out API family. Resolved: Anthropic `claude-sonnet-5`.**

Pinned in `configs/generation/generators.yaml`. Note on reproducibility: for the Claude
4.6 generation and later the dateless model id is itself a fixed snapshot, so it meets
the pinned-revision requirement in section 5. Budget capped at 20,000 documents with a
hard stop, and the family is used for R3 and R4 evaluation only. It never generates
training data, because it is held out.

Remaining open questions from section 10: item 3 (does smoothing help) and item 4
(reserve pool size). Neither blocks a phase.

### v1.3 (2026-08-31) - mined hard negatives are split at the cluster level

**Change.** Section 4 gains a rule for documents that enter training through mining.

Mined documents come from the reserve pool, which sits outside train/val/test and
therefore carries no split. Two obvious options are both wrong:

- *All mined documents to train.* There is then no held-out hard-negative set, and no way
  to tell whether mining generalized or simply memorized the examples it was given.
- *Split mined documents randomly.* Worse than useless. Documents within one failure
  cluster are near-identical by construction, so a random split places near-duplicates of
  the same failure on both sides. The held-out score then measures memorization while
  looking like generalization.

**Rule.** Mined failures are split by CLUSTER. Whole failure modes go to training; other
whole modes are held out untouched and never enter any training set. A model that
improves on held-out clusters has generalized to failure modes it has not seen. Under a
document-level split those two outcomes are indistinguishable.

Default holdout fraction: 0.25 of clusters, assigned by a deterministic hash of the
cluster id so successive rounds stay comparable. Held-out modes are not fixed this round;
they can be released into training in a later round and the mined ledger tracks that.

**Known limitation, recorded not hidden.** Proportional selection guarantees coverage of
CLUSTERS, not of true failure modes. If clustering merges two modes, the smaller one is
starved regardless of how fair the selection policy is, and nothing downstream can
detect it. Measured on a synthetic five-mode set with the offline hashing embedder: one
mode was absorbed into another and never selected, and one mode was split across two
clusters and over-weighted. `Atlas.quality_warnings()` therefore reports low silhouette,
dominant clusters, duplicate cluster labels and the use of a non-semantic embedder, so a
mining report cannot present clean-looking cluster shares without also showing why they
may be wrong.

### v1.4 (2026-08-31) - adversarial measurement and contamination thresholds

**1. Every attack is measured in two preprocessing conditions.**

FORGE's ingestion normalisation neutralises several attacks before the model sees the
text. Measured on the offline attack suite, `whitespace_perturb`, `zero_width_insert` and
`paragraph_insert`-style perturbations are removed or reduced by `normalize()`, while
`homoglyph_substitute` is NOT: Cyrillic look-alikes are distinct letters, not
compatibility forms, so NFKC leaves them untouched.

A single robustness number is therefore meaningless and wrong in a predictable direction:

    preprocessed only -> credits two lines of normalize() to the model; any deployment
                         that skips preprocessing is unprotected and the report hides it
    raw only          -> describes a threat the production path already handles

The report carries both columns and their difference, `preprocessing_benefit`, which
states the value of the preprocessing defence as a number instead of assuming it.

**2. Attack validity is checked with homoglyphs folded.**

A token-overlap validity check systematically misjudges character-level attacks. Cyrillic
substitutions are invisible to a reader but break every word containing them, so a
perfectly readable homoglyph attack scores a Jaccard near zero and would be discarded as
vandalism, filtering out precisely the attacks that work. `preserves_meaning()` folds
look-alikes back to Latin before comparing. That fold is for validity checking only and
never touches training or inference input.

**3. No-op attacks are excluded from scores, not counted as defences.**

A sparse-target attack such as `synonym_swap` genuinely changes nothing on a document
with no substitutable words. Scoring that as a successful defence reports the attack
lexicon's coverage as the model's robustness. No-ops are counted and reported separately.

**4. The contamination near-duplicate threshold is 0.5, looser than deduplication's 0.8.**

The error costs run in opposite directions:

    dedup false positive         -> two distinct documents merged, training data lost
    contamination false negative -> the external result is invalid and nobody knows

A benchmark document with a sentence of site boilerplate appended sits at a true Jaccard
of about 0.75, which 0.8 misses and 0.5 catches. Over-flagging costs a manual look;
under-flagging costs the headline claim. Reusing 0.8 here would inherit a threshold tuned
for the opposite trade.

**5. RAID variants are collapsed by `source_id` before scoring.** The twelve attacks are
applied to the same base generations, so scoring them independently counts one base text
up to thirteen times and lets a single unusually easy or hard text move the number.
