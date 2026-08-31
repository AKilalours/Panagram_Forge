"""Phase 1 orchestration: config in, FORGE-HUMAN out.

One design point worth stating. Deduplication is shared across all sources in a run,
not per source. FineWeb and FineWeb-Edu are both derived from Common Crawl, so the same
page can appear in both. Deduping per source would let that pair through, and the pair
would land in different splits, which is the leak this whole pipeline is built to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from forge.cleaning.pipeline import Cleaner, CleaningPolicy
from forge.common.config import REPO_ROOT, load
from forge.common.schemas import HumanDocument, Split
from forge.common.splits import check_no_group_leakage
from forge.ingestion import sources as src
from forge.ingestion.writer import write_manifest, write_metadata_only, write_parquet


@dataclass
class IngestResult:
    docs: list[HumanDocument]
    stats_by_source: dict[str, dict]
    partitions: dict[str, int]
    manifest_path: Path | None


def _build_source(entry: dict, local_root: Path | None):
    sid = entry["id"]
    if sid in ("fw", "fwe"):
        cls = src.FineWebSource if sid == "fw" else src.FineWebEduSource
        return cls(config=entry.get("config", "sample-10BT"), filters=entry.get("filters"))
    if sid == "gut":
        return src.GutenbergSource(entry.get("path") or (local_root or Path(".")) / "gutenberg")
    if sid == "gov":
        return src.GovInfoSource(entry.get("path") or (local_root or Path(".")) / "govinfo")
    if sid == "local":
        return src.LocalJSONLSource(
            entry["path"],
            source_id=entry.get("source_id", "local"),
            license=entry.get("license", "unknown"),
            domain=entry.get("domain", "web"),
            register=entry.get("register", "informational"),
        )
    raise ValueError(f"unknown source id {sid!r}")


def _quota(entry: dict, total: int | None) -> int | None:
    """Per-source document budget from its train_share."""
    if total is None:
        return None
    share = entry.get("train_share")
    return int(round(total * share)) if share else None


def ingest(config_path: str, out_root: str | Path | None = None, total: int | None = None) -> IngestResult:
    cfg = load(config_path)
    out_root = Path(out_root) if out_root else REPO_ROOT / "data" / "silver"
    total = total if total is not None else cfg.get("targets", {}).get("human_train")

    cleaning = cfg.get("cleaning", {})
    policy = CleaningPolicy()
    policy.language_score_min = cleaning.get("language_score_min", policy.language_score_min)
    policy.length.__dict__  # frozen dataclasses; defaults come from the spec

    # One cleaner for the whole run: dedup must be global across sources.
    cleaner = Cleaner(policy)
    all_docs: list[HumanDocument] = []
    stats_by_source: dict[str, dict] = {}

    for entry in cfg.get("sources", []):
        before_seen, before_kept = cleaner.stats.seen, cleaner.stats.kept
        before_rejected = cleaner.stats.rejected.copy()

        source = _build_source(entry, REPO_ROOT / "data" / "raw")
        quota = _quota(entry, total)
        # Read more than the quota because most documents are rejected.
        read_limit = quota * 20 if quota else None

        produced = 0
        for doc in cleaner.process(source.stream(limit=read_limit)):
            all_docs.append(doc)
            produced += 1
            if quota is not None and produced >= quota:
                break

        stats_by_source[entry["id"]] = {
            "seen": cleaner.stats.seen - before_seen,
            "kept": cleaner.stats.kept - before_kept,
            "rejected": dict(cleaner.stats.rejected - before_rejected),
        }

    # Hard invariant. If this raises, nothing downstream is trustworthy.
    check_no_group_leakage([(d.source_group_id, d.split) for d in all_docs])

    version = cfg.get("dataset_version", "v0.1")
    partitions = write_parquet(all_docs, out_root)
    # Publishable artifact lives outside the partition tree: different schema, and a
    # glob over the data lake must never pick it up.
    write_metadata_only(all_docs, REPO_ROOT / "data" / "publishable" / version / "metadata_only.parquet")
    manifest = write_manifest(out_root, version, all_docs, stats_by_source)
    return IngestResult(all_docs, stats_by_source, partitions, manifest)
