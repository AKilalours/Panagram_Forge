"""Parquet to windowed, tokenized training examples.

This is the join between the data lake and the model. It reads human documents, their
mirrors, and any mixed documents produced by token_labels.py, and emits fixed-width
windows with both a document label and per-token labels.

The one rule enforced here that nothing else can enforce: **a window's document label is
computed from its own content**, via modeling/dataset.py. A mostly-human document with one
AI paragraph must not stamp "ai" on its mostly-human windows.

torch is imported lazily so the rest of the package still imports without it.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from forge.common.schemas import Label, Split, TokenLabel
from forge.modeling.alignment import IGNORE_INDEX, align_spans_to_tokens
from forge.modeling.windowing import windows


@dataclass
class RawExample:
    doc_id: str
    source_group_id: str
    split: str
    text: str
    label: int                                  # document level, 0 human 1 ai
    spans: list[tuple[int, int, str]] | None    # character spans, None = single-label doc
    domain: str = "unknown"
    generator_family: str = "human"
    generator_released: str = ""


def _single_span(text: str, label: int) -> list[tuple[int, int, str]]:
    tl = TokenLabel.AI_GENERATED.value if label else TokenLabel.HUMAN.value
    return [(0, len(text), tl)]


ARM_PROMPT_VERSION = {"random": "random_v1", "mirror": "mirror_v1", "hard_negative": "mirror_v1"}


class ArmMismatch(RuntimeError):
    pass


def load_examples(
    human_root: str | Path | None = None,
    ai_root: str | Path | None = None,
    mixed_root: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    expect_arm: str | None = None,
    mirror_root: str | Path | None = None,   # deprecated alias
) -> list[RawExample]:
    """Load one arm's training data.

    `expect_arm` exists because of a bug that would have produced a FALSE NEGATIVE at
    real cost. All three arm configs pointed their AI source at the same directory, and
    `data.include` was decorative, so Arm A and Arm B would have trained on identical
    data, produced identical numbers, and supported the conclusion "matched mirrors make
    no difference".

    Nothing would have errored. Two successful runs, two plausible rows in the results
    table, one false finding. So the arm is now declared in the config and verified
    against the `prompt_version` recorded on every generated document.
    """
    ai_root = ai_root or mirror_root
    human_rows: list[RawExample] = []
    ai_rows: list[RawExample] = []
    mixed_rows: list[RawExample] = []

    if human_root:
        from forge.ingestion.writer import PARTITION_GLOB

        for f in sorted(glob.glob(str(Path(human_root) / PARTITION_GLOB))):
            for r in pq.read_table(
                f, columns=["doc_id", "source_group_id", "text", "split", "domain"]
            ).to_pylist():
                if r["split"] not in splits:
                    continue
                human_rows.append(RawExample(r["doc_id"], r["source_group_id"], r["split"],
                                             r["text"], 0, _single_span(r["text"], 0), r["domain"]))

    if ai_root:
        seen_versions: set[str] = set()
        for f in sorted(glob.glob(str(Path(ai_root) / "split=*/*.parquet"))):
            for r in pq.read_table(
                f, columns=["sample_id", "source_group_id", "text", "split", "domain",
                            "generator", "mirror"]
            ).to_pylist():
                if r["split"] not in splits:
                    continue
                g = r.get("generator") or {}
                seen_versions.add(((r.get("mirror") or {}).get("prompt_version")) or "unknown")
                ai_rows.append(RawExample(r["sample_id"], r["source_group_id"], r["split"],
                                          r["text"], 1, _single_span(r["text"], 1), r["domain"],
                                          g.get("family", "unknown"), str(g.get("released", ""))))

        if expect_arm and seen_versions:
            want = ARM_PROMPT_VERSION.get(expect_arm)
            if want and not seen_versions <= {want}:
                raise ArmMismatch(
                    f"config declares arm '{expect_arm}' (expects prompt_version "
                    f"{want!r}) but {ai_root} contains {sorted(seen_versions)}. The arm "
                    "is training on the wrong data. Two arms reading the same directory "
                    "produce identical results and a false finding."
                )

    if mixed_root:
        for f in sorted(glob.glob(str(Path(mixed_root) / "split=*/*.parquet"))):
            for r in pq.read_table(f).to_pylist():
                if r["split"] not in splits:
                    continue
                spans = [(s["start_char"], s["end_char"], s["label"]) for s in r["spans"]]
                ai = sum(e - s for s, e, l in spans if l != TokenLabel.HUMAN.value)
                mixed_rows.append(RawExample(r["doc_id"], r["source_group_id"], r["split"], r["text"],
                                             int(ai / max(len(r["text"]), 1) >= 0.5), spans,
                                             r.get("domain", "unknown")))

    if limit is None:
        return human_rows + ai_rows + mixed_rows
    return interleave(*_by_source_and_split(human_rows, ai_rows, mixed_rows), limit=limit)


def _by_source_and_split(*sources: list[RawExample]) -> list[list[RawExample]]:
    """One bucket per (source, split) pair, so a limit spans both.

    Round-robining over sources alone is not enough. The parquet files are partitioned by
    split and read in sorted order, so the front of each source's list is a single split:
    a 200-row limit drew 100 humans and 100 AI documents that were all `test`, leaving no
    training windows at all. Splitting the buckets further means a limit always contains
    train, val and test.
    """
    buckets: list[list[RawExample]] = []
    for rows in sources:
        grouped: dict[str, list[RawExample]] = {}
        for row in rows:
            grouped.setdefault(row.split, []).append(row)
        buckets.extend(grouped[key] for key in sorted(grouped))
    return buckets


def interleave(*buckets: list[RawExample], limit: int | None = None) -> list[RawExample]:
    """Concatenate the sources, or take a limit that draws from ALL of them.

    WHY. Each loader used to append to one list and return early once it had `limit`
    rows. Humans are read first, so any limited load returned humans only: `--smoke`,
    which is --limit 200, trained on 200 human documents and zero AI documents. The run
    completed, reported a loss, and told you nothing, because a classifier fed one class
    cannot fail in a way a smoke test would notice.

    That is the same defect as the corpus loader's prefix truncation, in a different
    file. The lesson worth keeping: an early return inside a source-specific loop is
    never a subsample, it is a filter on source.

    Round-robin across the buckets so a limit is a real sample of what a full run would
    see. Callers pass one bucket per (source, split) pair; see _by_source_and_split.
    """
    if limit is None:
        return [row for bucket in buckets for row in bucket]
    picked: list[RawExample] = []
    index = 0
    while len(picked) < limit and any(index < len(b) for b in buckets):
        for bucket in buckets:
            if index < len(bucket):
                picked.append(bucket[index])
                if len(picked) >= limit:
                    break
        index += 1
    return picked


def build_dataset(examples: list[RawExample], tokenizer, max_length: int = 512,
                  stride: int = 384, ai_window_threshold: float = 0.5):
    """Tokenize once per document, then slice into overlapping windows.

    Tokenizing per window instead would retokenize overlapping text and, worse, would put
    window boundaries at token positions that differ from the offsets used for span
    alignment.
    """
    import torch

    feats = []
    for ex in examples:
        enc = tokenizer(ex.text, return_offsets_mapping=True, add_special_tokens=False,
                        truncation=False)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        ids = enc["input_ids"]
        if not ids:
            continue
        spans = [(s, e, TokenLabel(l)) for s, e, l in (ex.spans or _single_span(ex.text, ex.label))]
        token_labels = align_spans_to_tokens(offsets, spans)

        # reserve two positions for the special tokens the collator adds
        body = max_length - 2
        for wi, (a, b) in enumerate(windows(len(ids), size=body, stride=min(stride, body))):
            wl = token_labels[a:b]
            real = [x for x in wl if x != IGNORE_INDEX]
            frac = (sum(1 for x in real if x != 0) / len(real)) if real else 0.0
            feats.append({
                "input_ids": ids[a:b],
                "token_labels": wl,
                "doc_label": int(frac >= ai_window_threshold),
                "doc_id": ex.doc_id,
                "source_group_id": ex.source_group_id,
                "split": ex.split,
                "domain": ex.domain,
                "generator_family": ex.generator_family,
                "window_index": wi,
                "ai_char_fraction": frac,
            })
    return feats


class Collator:
    """Pads a batch and adds special tokens, keeping token labels aligned.

    Special tokens get IGNORE_INDEX so the token head is never trained to predict
    authorship for a [CLS] that carries no text.
    """

    def __init__(self, tokenizer, max_length: int = 512) -> None:
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        import torch

        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        if pad is None:
            pad = 0
        ids, labs = [], []
        for b in batch:
            seq = ([cls] if cls is not None else []) + list(b["input_ids"]) + ([sep] if sep is not None else [])
            lab = ([IGNORE_INDEX] if cls is not None else []) + list(b["token_labels"]) + ([IGNORE_INDEX] if sep is not None else [])
            ids.append(seq[: self.max_length])
            labs.append(lab[: self.max_length])
        width = max(len(x) for x in ids)
        attn = [[1] * len(x) + [0] * (width - len(x)) for x in ids]
        ids = [x + [pad] * (width - len(x)) for x in ids]
        labs = [x + [IGNORE_INDEX] * (width - len(x)) for x in labs]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "token_labels": torch.tensor(labs, dtype=torch.long),
            "doc_labels": torch.tensor([b["doc_label"] for b in batch], dtype=torch.long),
        }
