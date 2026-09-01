"""Arm A: conventional random synthetic data. The control the whole project measures against.

This is the naive approach the research question is testing MIRRORS against: pick a topic,
ask a model to write about it, label it AI. No matching to any human document.

    mirror:  human doc -> extract topic/genre/length/structure -> prompt -> AI doc
    random:  topic list -> "write an article about X" -> AI doc

Getting this arm RIGHT is what makes the comparison honest, and there are three ways to
get it wrong that all flatter the mirror arm:

**Equal budget.** Arm A must produce the SAME number of AI documents as Arm B, from the
SAME generator families, over the SAME decoding grid. If mirrors win on volume or on
generator diversity, the result says nothing about matching.

**A fair prompt.** The random prompt has to be what a competent person would actually
write, not a strawman. If Arm A's prompts are deliberately poor, the comparison is
rigged. The template below asks for a specific genre and length, which is what anyone
building a detector this way would do.

**Realistic length distribution.** Random generation still has to produce documents in
the same length RANGE as the corpus, or the detector learns length and Arm A collapses
for a reason unrelated to topic matching. What it must NOT do is match any individual
human document's length, because per-document matching is precisely what mirrors add.

Topics are drawn from a fixed inventory, never from the human corpus. Sampling topics
from the corpus would make this a weak mirror rather than a random baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.common.schemas import GeneratorSpec, Label, MirrorSpec, Split, SyntheticDocument
from forge.common.splits import assign_split
from forge.generation.assignment import (
    assert_no_held_out,
    assign_decoding,
    assign_family,
    held_in_families,
    parse_roster,
)
from forge.generation.generators.base import Decoding
from forge.generation.mirror import ValidationStats, strip_wrapper

# Deliberately broad and mundane. Not derived from FineWeb, and not curated to be
# difficult. This is what a topic list looks like when someone builds it in an afternoon.
TOPICS = [
    "coastal erosion", "municipal budgeting", "beekeeping", "railway signalling",
    "medieval bookbinding", "soil chemistry", "harbour dredging", "choral notation",
    "textile dyeing", "glacier monitoring", "orchard grafting", "lighthouse keeping",
    "urban composting", "ferry timetabling", "dry stone walling", "kiln firing",
    "salt marsh restoration", "clock repair", "cheese ripening", "canal locks",
    "seed banking", "bell ringing", "peat cutting", "thatching", "wool spinning",
    "tidal energy", "archive cataloguing", "hedgerow management", "brick making",
    "well drilling", "sail making", "grain milling", "lace making", "quarrying",
    "rope walking", "charcoal burning", "reed cutting", "basket weaving",
    "school timetabling", "library lending", "waste collection", "street lighting",
    "flood defences", "cycle infrastructure", "allotment policy", "park maintenance",
]
GENRES = [
    "explainer", "formal report", "how-to guide", "historical overview",
    "opinion column", "encyclopedia entry", "news article", "review",
]
REGISTERS = ["informational", "formal", "conversational"]

PROMPT = """Write a {genre} in a {register} register about {topic}.

- Length: approximately {target_tokens} tokens.
- Write continuous prose unless the genre calls for another shape.

Output only the text itself. Do not add a preamble, a title line, a sign-off, or any
meta commentary."""


@dataclass
class RandomSpec:
    topic: str
    genre: str
    register: str
    target_tokens: int

    def prompt(self) -> str:
        return PROMPT.format(genre=self.genre, register=self.register, topic=self.topic,
                             target_tokens=self.target_tokens)


def _digest(key: str, salt: str) -> int:
    return int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest()[:12], 16)


def spec_for(index: int, length_pool: list[int]) -> RandomSpec:
    """Deterministic, so a partial run resumes to the same dataset.

    `length_pool` is the observed token-count distribution of the human corpus. Sampling
    a length from that pool keeps Arm A in the same length RANGE as Arm B without
    matching any individual document, which is the distinction the experiment turns on.
    """
    k = str(index)
    return RandomSpec(
        topic=TOPICS[_digest(k, "topic") % len(TOPICS)],
        genre=GENRES[_digest(k, "genre") % len(GENRES)],
        register=REGISTERS[_digest(k, "register") % len(REGISTERS)],
        target_tokens=length_pool[_digest(k, "len") % len(length_pool)],
    )


@dataclass
class RandomResult:
    docs: list[SyntheticDocument] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def generate_random(
    n: int,
    length_pool: list[int],
    generators_cfg: dict,
    backend: str = "fake",
    length_ratio_min: float = 0.5,
    length_ratio_max: float = 2.0,
    max_retries: int = 2,
) -> RandomResult:
    from forge.generation.mirror import _PREAMBLE
    from forge.generation.run import build_generator

    if not length_pool:
        raise ValueError("length_pool is empty; read it from the human corpus")

    roster = parse_roster(generators_cfg)
    families = held_in_families(roster)
    grid = generators_cfg.get("decoding_grid", {})
    cache: dict[str, object] = {}
    used: set[str] = set()
    stats = ValidationStats()
    out: list[SyntheticDocument] = []
    now = datetime.now(timezone.utc)

    for i in range(n):
        spec = spec_for(i, length_pool)
        doc_key = f"rand_{i}"
        fam = assign_family(doc_key, families)
        used.add(fam.family)
        gen = cache.setdefault(fam.family, build_generator(fam, backend))
        decoding: Decoding = assign_decoding(doc_key, grid)

        accepted = None
        for attempt in range(max_retries + 1):
            d = Decoding(decoding.temperature, decoding.top_p, decoding.max_new_tokens,
                         (decoding.seed or 0) + attempt)
            text = strip_wrapper(gen.generate([spec.prompt()], d)[0])
            if not text:
                stats.reject("empty"); continue
            if _PREAMBLE.match(text):
                stats.reject("assistant_preamble"); continue
            ratio = len(text.split()) / max(spec.target_tokens, 1)
            if ratio < length_ratio_min:
                stats.reject("too_short"); continue
            if ratio > length_ratio_max:
                stats.reject("too_long"); continue
            accepted = (text, d); break
        if accepted is None:
            continue

        text, d = accepted
        stats.accepted += 1
        # No human source, so this document gets its own group and its own split.
        group = f"grp_rand_{i}"
        out.append(SyntheticDocument(
            sample_id=f"rand_{i}", source_human_id=f"none_rand_{i}", source_group_id=group,
            label=Label.AI, text=text, split=assign_split(group),
            generator=GeneratorSpec(
                provider="api" if fam.provider == "api" else "open_source",
                family=fam.family, model_id=getattr(gen, "model_id", fam.model_id),
                revision=getattr(gen, "revision", fam.revision),
                temperature=d.temperature, top_p=d.top_p,
                max_new_tokens=d.max_new_tokens, seed=d.seed,
            ),
            mirror=MirrorSpec(
                prompt_version="random_v1", target_tokens=spec.target_tokens,
                # Explicitly false. This is the control: nothing is matched.
                topic_match=False, length_match=False, style_match=False,
                attributes={"topic": spec.topic, "genre": spec.genre,
                            "register": spec.register, "arm": "random", "backend": backend},
            ),
            domain="web", generated_at=now,
        ))

    assert_no_held_out(roster, used)
    s = stats.as_dict()
    s.update(families_used=sorted(used), backend=backend, arm="random")
    return RandomResult(out, s)


def length_pool_from_corpus(human_root: str | Path) -> list[int]:
    """Token counts of the human corpus, so Arm A matches its length DISTRIBUTION."""
    import glob

    import pyarrow.parquet as pq

    from forge.ingestion.writer import PARTITION_GLOB

    pool = [
        r["token_count"]
        for f in sorted(glob.glob(str(Path(human_root) / PARTITION_GLOB)))
        for r in pq.read_table(f, columns=["token_count"]).to_pylist()
    ]
    if not pool:
        raise FileNotFoundError(f"no human corpus under {human_root}; run `forge ingest` first")
    return pool
