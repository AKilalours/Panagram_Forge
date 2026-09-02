"""Phase 2 orchestration: human documents in, FORGE-MIRROR out.

The single most important line in this file is the one that copies `source_group_id`
and `split` from the human document onto its mirror. A mirror must never be assigned a
split of its own. If it were, a human document could land in train while its mirror
lands in test, and since the two share topic, length and structure, the model would
score near-perfectly on content it had already memorized. Every metric would improve and
nothing would raise an error.

Second most important: `assert_no_held_out` runs before anything is written. A held-out
family generating training data invalidates R3, the unseen-generator regime, which is
the headline claim of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from forge.common.config import REPO_ROOT, load
from forge.common.schemas import GeneratorSpec, MirrorSpec, Split, SyntheticDocument
from forge.dedup.minhash import MinHash
from forge.generation.assignment import (
    FamilySpec,
    assert_no_held_out,
    assign_decoding,
    assign_family,
    held_in_families,
    parse_roster,
)
from forge.generation.attributes import HeuristicExtractor, VerbatimCopyError
from forge.generation.generators.base import (
    APIGenerator,
    Decoding,
    FakeGenerator,
    TransformersGenerator,
    VLLMGenerator,
)
from forge.generation.mirror import ValidationPolicy, ValidationStats, load_template, render_prompt, strip_wrapper, validate

PARTITION_GLOB = "split=*/*.parquet"


@dataclass
class HumanRef:
    """The minimum a mirror needs from its human source. Deliberately not the full
    HumanDocument, so this runner can consume Parquet rows without a schema round-trip."""

    doc_id: str
    source_group_id: str
    text: str
    domain: str
    split: Split


@dataclass
class MirrorResult:
    docs: list[SyntheticDocument]
    stats: dict = field(default_factory=dict)
    partitions: dict[str, int] = field(default_factory=dict)


# How many prompts to hand the backend at once.
#
# vLLM schedules internally, so this is not the batch size the GPU sees; it bounds how
# much output is held in Python at a time and gives the run a progress signal. Too small
# and the engine idles between calls, too large and a crash loses more work.
GENERATION_BATCH = 512


@dataclass
class _Work:
    """One document's worth of prepared, model-free work."""

    human: HumanRef
    attrs: object
    spec: FamilySpec
    decoding: Decoding
    prompt: str


def build_generator(spec: FamilySpec, backend: str):
    if backend == "fake":
        return FakeGenerator(family=spec.family, model_id=spec.model_id, revision="fake")
    if spec.provider == "api":
        return APIGenerator(spec.family, spec.model_id, endpoint="")
    if backend == "vllm":
        return VLLMGenerator(spec.family, spec.model_id, spec.revision)
    if backend == "transformers":
        return TransformersGenerator(spec.family, spec.model_id, spec.revision)
    raise ValueError(f"unknown backend {backend!r}")


def generate_mirrors(
    humans: list[HumanRef],
    generators_cfg: dict,
    mirror_cfg: dict,
    backend: str = "fake",
) -> MirrorResult:
    roster = parse_roster(generators_cfg)
    families = held_in_families(roster)
    grid = generators_cfg.get("decoding_grid", {})

    policy_cfg = mirror_cfg.get("validation", {})
    policy = ValidationPolicy(
        length_ratio_min=policy_cfg.get("length_ratio_min", 0.6),
        length_ratio_max=policy_cfg.get("length_ratio_max", 1.6),
        reject_assistant_preamble=policy_cfg.get("reject_assistant_preamble", True),
        max_minhash_jaccard_to_source=policy_cfg.get("max_minhash_jaccard_to_source", 0.5),
        max_retries=policy_cfg.get("max_retries", 2),
    )
    template = load_template(REPO_ROOT / mirror_cfg["prompt_path"])
    prompt_version = mirror_cfg.get("prompt_version", "mirror_v1")

    extractor = HeuristicExtractor()
    hasher = MinHash()
    stats = ValidationStats()
    extraction_failures = 0
    used_families: set[str] = set()

    # PASS 1. Extraction, family assignment, decoding assignment and prompt rendering are
    # pure CPU and need no model loaded. Doing them all first is what lets pass 2 hold
    # exactly one engine in memory at a time.
    work: list[_Work] = []
    for human in humans:
        try:
            attrs = extractor.extract(human.text)
        except VerbatimCopyError:
            extraction_failures += 1
            continue
        spec = assign_family(human.doc_id, families)
        used_families.add(spec.family)
        work.append(
            _Work(
                human=human,
                attrs=attrs,
                spec=spec,
                decoding=assign_decoding(human.doc_id, grid),
                prompt=render_prompt(attrs, template),
            )
        )

    by_family: dict[str, list[_Work]] = {}
    for w in work:
        by_family.setdefault(w.spec.family, []).append(w)

    # PASS 2. One family at a time, in large batches, releasing before the next.
    #
    # This replaces a loop that walked documents in order, kept a generator per family in
    # a cache, and called generate([single_prompt]) once per document. That shape had two
    # defects that only appear on real hardware:
    #
    #   1. Documents are assigned to families round-robin, so the cache held every
    #      family's engine at once. A vLLM engine reserves ~90% of the card at startup, so
    #      the second engine found 1 GiB free and the run died with "Free memory on device
    #      (1.03/23.53 GiB) ... is less than desired GPU memory utilization".
    #   2. One prompt per call never engages continuous batching, which is the only reason
    #      vLLM is used here. At roughly 8 seconds per document, 60k documents is over 130
    #      hours: not slow, infeasible.
    #
    # Grouping by family fixes both at once. Assignment is unchanged, so which family
    # generates which document is byte-identical to before; only the order of work moves.
    accepted: dict[str, tuple[str, Decoding]] = {}
    identity: dict[str, tuple[str, str, str]] = {}
    for family in sorted(by_family):
        items = by_family[family]
        gen = build_generator(items[0].spec, backend)
        # Capture identity BEFORE close(), which drops the engine the attributes hang off.
        identity[family] = (
            getattr(gen, "model_id", items[0].spec.model_id),
            getattr(gen, "revision", items[0].spec.revision),
            items[0].spec.provider,
        )
        print(f"[mirror] family={family} documents={len(items)} backend={backend}", flush=True)
        try:
            pending = items
            for attempt in range(policy.max_retries + 1):
                if not pending:
                    break
                # Vary the seed per attempt, otherwise a deterministic backend returns the
                # identical rejected text and the retries are pure waste.
                decs = [
                    Decoding(
                        w.decoding.temperature,
                        w.decoding.top_p,
                        w.decoding.max_new_tokens,
                        (w.decoding.seed or 0) + attempt,
                    )
                    for w in pending
                ]
                prompts = [w.prompt + (" " * attempt) for w in pending]
                texts: list[str] = []
                for i in range(0, len(prompts), GENERATION_BATCH):
                    texts.extend(
                        gen.generate_many(
                            prompts[i : i + GENERATION_BATCH], decs[i : i + GENERATION_BATCH]
                        )
                    )
                    done = min(i + GENERATION_BATCH, len(prompts))
                    print(
                        f"[mirror] family={family} attempt={attempt} {done}/{len(prompts)}",
                        flush=True,
                    )
                still: list[_Work] = []
                for w, d, raw in zip(pending, decs, texts):
                    text = strip_wrapper(raw)
                    ok, reason = validate(text, w.human.text, w.attrs, policy, _hasher=hasher)
                    if ok:
                        accepted[w.human.doc_id] = (text, d)
                        stats.accepted += 1
                    else:
                        stats.reject(reason)
                        still.append(w)
                pending = still
        finally:
            gen.close()

    # PASS 3. Assemble in the humans' original order, so the parquet output does not
    # depend on which family happened to be processed first.
    out: list[SyntheticDocument] = []
    for w in work:
        got = accepted.get(w.human.doc_id)
        if got is None:
            continue
        text, d = got
        model_id, revision, provider = identity[w.spec.family]
        human, attrs = w.human, w.attrs
        out.append(
            SyntheticDocument(
                sample_id=f"forge_{human.doc_id}",
                source_human_id=human.doc_id,
                # Inherited, never reassigned. See the module docstring.
                source_group_id=human.source_group_id,
                split=human.split,
                text=text,
                generator=GeneratorSpec(
                    provider="api" if provider == "api" else "open_source",
                    family=w.spec.family,
                    model_id=model_id,
                    revision=revision,
                    temperature=d.temperature,
                    top_p=d.top_p,
                    max_new_tokens=d.max_new_tokens,
                    seed=d.seed,
                ),
                mirror=MirrorSpec(
                    prompt_version=prompt_version,
                    target_tokens=attrs.target_tokens,
                    topic_match=True,
                    length_match=True,
                    style_match=True,
                    attributes={
                        "genre": attrs.genre,
                        "register": attrs.register,
                        "structure": attrs.structure,
                        "difficulty": attrs.difficulty,
                        "extractor": extractor.name,
                        "backend": backend,
                    },
                ),
                domain=human.domain,
                generated_at=datetime.now(timezone.utc),
            )
        )

    # Hard checks before anything is written.
    assert_no_held_out(roster, used_families)
    _assert_split_inheritance(humans, out)

    s = stats.as_dict()
    s["extraction_failures"] = extraction_failures
    s["families_used"] = sorted(used_families)
    s["backend"] = backend
    return MirrorResult(out, s)


def _assert_split_inheritance(humans: list[HumanRef], mirrors: list[SyntheticDocument]) -> None:
    by_id = {h.doc_id: h for h in humans}
    for m in mirrors:
        h = by_id[m.source_human_id]
        if m.split is not h.split or m.source_group_id != h.source_group_id:
            raise RuntimeError(
                f"mirror {m.sample_id} did not inherit its human's split/group. "
                "This is the leak the whole split design exists to prevent."
            )


def write_mirrors(docs: list[SyntheticDocument], root: str | Path) -> dict[str, int]:
    root = Path(root)
    written: dict[str, int] = {}
    groups: dict[str, list[dict]] = {}
    for d in docs:
        row = d.model_dump(mode="json")
        row["generator"] = row["generator"]  # struct column, pyarrow infers it
        groups.setdefault(d.split.value, []).append(row)
    for split, rows in groups.items():
        out = root / f"split={split}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), out / "part-000.parquet", compression="zstd")
        written[split] = len(rows)
    return written


def read_humans(lake_root: str | Path, limit: int | None = None) -> list[HumanRef]:
    import glob

    from forge.ingestion.writer import PARTITION_GLOB as HUMAN_GLOB

    files = sorted(glob.glob(str(Path(lake_root) / HUMAN_GLOB)))
    if not files:
        raise FileNotFoundError(f"no human parquet under {lake_root}. Run `forge ingest` first.")
    refs: list[HumanRef] = []
    for f in files:
        t = pq.read_table(f, columns=["doc_id", "source_group_id", "text", "domain", "split"])
        for row in t.to_pylist():
            refs.append(HumanRef(row["doc_id"], row["source_group_id"], row["text"], row["domain"], Split(row["split"])))
            if limit is not None and len(refs) >= limit:
                return refs
    return refs


DEFAULT_GENERATORS_CONFIG = "configs/generation/generators.yaml"


def resolve_generators_config(mirror_cfg: dict) -> str:
    """Which generator roster does this mirror config name?

    This path used to be hardcoded to the full-scale roster, so running the
    minimal mirror config silently generated from a DIFFERENT set of model
    families than the one it declared, at unpinned revisions. Nothing errored;
    the run was only stopped further downstream by require_pinned_revision,
    which exists for a different reason and happened to catch it.

    A config key that the code never reads is indistinguishable from a typo.
    Read it here, and fail loudly if it names a file that does not exist rather
    than falling back to the default, because a silent fallback reintroduces
    exactly the bug this replaces.
    """
    named = mirror_cfg.get("generators_config")
    if named is None:
        return DEFAULT_GENERATORS_CONFIG
    if not isinstance(named, str) or not named.strip():
        raise ValueError(f"generators_config must be a non-empty path, got {named!r}")
    if not (REPO_ROOT / named).is_file():
        raise FileNotFoundError(
            f"mirror config names generators_config={named!r}, which does not exist. "
            "Fix the path rather than letting it fall back to the default roster."
        )
    return named


def run(config_path: str, humans_root: str | Path, out_root: str | Path, backend: str = "fake", limit: int | None = None) -> MirrorResult:
    mirror_cfg = load(config_path)
    generators_cfg = load(resolve_generators_config(mirror_cfg))
    humans = read_humans(humans_root, limit=limit)
    result = generate_mirrors(humans, generators_cfg, mirror_cfg, backend=backend)
    result.partitions = write_mirrors(result.docs, out_root)
    return result
