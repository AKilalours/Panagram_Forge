"""External benchmark loaders. EVAL-ONLY, always.

Schemas below were read from the live dataset cards on 2026-08-31 and are recorded in
data_spec_v1 section 1.1. They are pinned here as constants so a silent upstream schema
change fails loudly at load time instead of producing a plausible wrong number.

Nothing in this file has been run against the real datasets. Neither the development
sandbox nor the cloud container can reach huggingface.co, so every loader is verified
against synthetic fixtures matching the recorded schema exactly. The parsing logic is
tested; the download is not.

Three per-dataset traps, each of which produces a believable but wrong result:

  MAGE polarity. The card says label 1 means machine-generated; the original
  DeepfakeTextDetect release used the opposite convention in places. An inverted label
  turns an AUROC of 0.05 into a reported 0.95 and nothing crashes. Asserted at load.

  HC3 shape. `human_answers` and `chatgpt_answers` are LISTS. One row expands to several
  documents, so row counts are not document counts, and every document from one question
  must share a group id or the same question appears on both sides of any split.

  RAID adversarial grouping. Twelve attacks are applied to the same base generations, and
  they share `source_id`. Scoring them as independent documents counts one base text up
  to thirteen times and lets a single easy or hard base text dominate the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

# Pinned schemas. A mismatch is a hard error, not a warning.
RAID_COLUMNS = {
    "id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty",
    "attack", "domain", "title", "prompt", "generation",
}
MAGE_COLUMNS = {"text", "label", "src"}
HC3_COLUMNS = {"id", "question", "human_answers", "chatgpt_answers", "source"}

# English natural language only. FORGE v1 is not a code or multilingual detector, and
# scoring it on those inflates or deflates the number for reasons unrelated to the
# research question. Reported separately, never in the headline.
RAID_EXCLUDED_DOMAINS = frozenset({"code", "czech", "german"})

# RAID's human rows carry model == "human".
RAID_HUMAN_MODEL = "human"


class SchemaMismatch(RuntimeError):
    pass


class LabelPolarityError(AssertionError):
    pass


@dataclass
class BenchmarkDoc:
    """One evaluation document, normalised across the three benchmarks."""

    doc_id: str
    group_id: str          # shared by variants of the same base text
    text: str
    label: int             # 1 = machine-generated, FORGE's convention throughout
    benchmark: str
    domain: str = "unknown"
    model: str = "unknown"
    attack: str = "none"
    meta: dict = field(default_factory=dict)


def _require_columns(present: Iterable[str], expected: set[str], name: str) -> None:
    present = set(present)
    missing = expected - present
    if missing:
        raise SchemaMismatch(
            f"{name}: expected columns {sorted(missing)} are absent. The upstream schema "
            f"changed since data_spec_v1 section 1.1 was written (saw {sorted(present)}). "
            "Update the spec and this loader deliberately rather than coercing."
        )


# --------------------------------------------------------------------------- RAID

def parse_raid(
    rows: Iterable[dict],
    excluded_domains: frozenset[str] = RAID_EXCLUDED_DOMAINS,
    include_attacks: bool = True,
) -> list[BenchmarkDoc]:
    rows = list(rows)
    if rows:
        _require_columns(rows[0].keys(), RAID_COLUMNS, "RAID")
    out: list[BenchmarkDoc] = []
    for r in rows:
        domain = str(r.get("domain", "")).lower()
        if domain in excluded_domains:
            continue
        attack = str(r.get("attack") or "none")
        if not include_attacks and attack != "none":
            continue
        model = str(r.get("model", ""))
        out.append(
            BenchmarkDoc(
                doc_id=str(r["id"]),
                # Attacked variants share the base generation's source_id. Grouping by it
                # stops one base text being counted up to thirteen times.
                group_id=str(r.get("source_id") or r["id"]),
                text=str(r.get("generation", "")),
                label=0 if model == RAID_HUMAN_MODEL else 1,
                benchmark="raid",
                domain=domain or "unknown",
                model=model or "unknown",
                attack=attack,
                meta={
                    "decoding": r.get("decoding"),
                    "repetition_penalty": r.get("repetition_penalty"),
                    "adv_source_id": r.get("adv_source_id"),
                },
            )
        )
    return out


# --------------------------------------------------------------------------- MAGE

def parse_mage(rows: Iterable[dict], machine_label: int = 1) -> list[BenchmarkDoc]:
    rows = list(rows)
    if rows:
        _require_columns(rows[0].keys(), MAGE_COLUMNS, "MAGE")
    return [
        BenchmarkDoc(
            doc_id=f"mage_{i}",
            group_id=f"mage_{i}",
            text=str(r["text"]),
            label=1 if int(r["label"]) == machine_label else 0,
            benchmark="mage",
            model=str(r.get("src", "unknown")),
            meta={"src": r.get("src")},
        )
        for i, r in enumerate(rows)
    ]


def assert_mage_polarity(docs: list[BenchmarkDoc], scores: list[float], margin: float = 0.05) -> None:
    """Fail loudly if the benchmark's labels are inverted relative to FORGE's convention.

    Uses the detector's own scores: documents labelled machine must score higher on
    average than documents labelled human. If they do not, either the labels are inverted
    or the detector is worse than random, and both must stop the run rather than produce
    a number.
    """
    y = np.asarray([d.label for d in docs], int)
    s = np.asarray(scores, float)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise LabelPolarityError("cannot check polarity without both classes present")
    m, h = float(s[y == 1].mean()), float(s[y == 0].mean())
    if m <= h + margin:
        raise LabelPolarityError(
            f"MAGE polarity check failed: mean score on machine-labelled documents "
            f"({m:.4f}) is not above human-labelled ({h:.4f}) by {margin}. Either the "
            "label convention is inverted or the detector is near-random. Both invalidate "
            "the reported number."
        )


# --------------------------------------------------------------------------- HC3

def parse_hc3(rows: Iterable[dict]) -> list[BenchmarkDoc]:
    """One row expands to several documents. Every document from one question shares a
    group id, so a question never lands on both sides of any split or resample."""
    rows = list(rows)
    if rows:
        _require_columns(rows[0].keys(), HC3_COLUMNS, "HC3")
    out: list[BenchmarkDoc] = []
    for r in rows:
        qid = str(r.get("id", len(out)))
        group = f"hc3_q{qid}"
        for kind, answers, label in (
            ("human", r.get("human_answers") or [], 0),
            ("chatgpt", r.get("chatgpt_answers") or [], 1),
        ):
            for j, ans in enumerate(answers):
                text = str(ans).strip()
                if not text:
                    continue
                out.append(
                    BenchmarkDoc(
                        doc_id=f"{group}_{kind}_{j}",
                        group_id=group,
                        text=text,
                        label=label,
                        benchmark="hc3",
                        domain=str(r.get("source", "unknown")),
                        model="chatgpt" if label else "human",
                        meta={"question": r.get("question")},
                    )
                )
    return out


# --------------------------------------------------------------------------- loading

def load_hf(repo: str, config: str | None, split: str) -> Iterator[dict]:  # pragma: no cover
    """Streaming load. Not exercised: no network in the development environment."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("needs the `data` extra: pip install -e '.[data]'") from e
    yield from load_dataset(repo, name=config, split=split, streaming=True)


def load_raid(split: str = "extra") -> list[BenchmarkDoc]:  # pragma: no cover
    """RAID-extra: labelled and disjoint from RAID-train. RAID-test labels are held out,
    so a self-reported number on it is impossible; that needs a leaderboard submission."""
    if split == "test":
        raise ValueError(
            "RAID-test labels are held out. Self-reporting on it is impossible; "
            "submit to the leaderboard instead. See data_spec_v1 section 1.1."
        )
    return parse_raid(load_hf("liamdugan/raid", None, split))


def load_mage(split: str = "test") -> list[BenchmarkDoc]:  # pragma: no cover
    return parse_mage(load_hf("yaful/MAGE", None, split))


def load_hc3() -> list[BenchmarkDoc]:  # pragma: no cover
    return parse_hc3(load_hf("Hello-SimpleAI/HC3", "all", "train"))


def group_aware_scores(docs: list[BenchmarkDoc], scores: list[float]) -> tuple[list[int], list[float]]:
    """Collapse variants of the same base text to one point, averaging their scores.

    Without this, RAID's twelve attacks make one base generation count thirteen times, so
    a single unusually easy or hard base text moves the headline number.
    """
    agg: dict[str, list[float]] = {}
    lab: dict[str, int] = {}
    for d, s in zip(docs, scores):
        agg.setdefault(d.group_id, []).append(float(s))
        lab[d.group_id] = d.label
    keys = sorted(agg)
    return [lab[k] for k in keys], [float(np.mean(agg[k])) for k in keys]
