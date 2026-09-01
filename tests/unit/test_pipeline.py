"""End-to-end cleaning pipeline behaviour, including the ordering properties that are
correctness rather than style."""

from datetime import datetime, timezone

from forge.cleaning.pipeline import Cleaner, run
from forge.common.splits import check_no_group_leakage
from forge.ingestion.sources import RawRecord

PROSE = (
    "The harbour authority published its annual dredging schedule in March, listing "
    "eleven separate operations across the estuary and the two tidal basins. Silt "
    "accumulation had increased noticeably since the previous survey, and the deeper "
    "berths required attention before the autumn shipping season began in earnest. "
    "Contractors were appointed in two lots, with the smaller basins handled under a "
    "framework agreement that had been renewed the year before. A short consultation "
    "period followed, during which several fishing associations objected to the "
    "proposed timing on the grounds that it overlapped with the spawning season."
)


def rec(rid: str, text: str, license: str = "unknown") -> RawRecord:
    return RawRecord(
        source_id="t", source_record_id=rid, text=text, license=license,
        domain="web", register="informational", acquired_at=datetime.now(timezone.utc),
    )


def test_clean_document_survives():
    docs, stats = run([rec("a", PROSE)])
    assert len(docs) == 1 and stats.kept == 1
    assert docs[0].doc_id == "t_a"
    assert docs[0].source_group_id == "grp_t_a"


def test_short_document_is_rejected_for_length_not_language():
    """Regression guard for the spec v1.1 stage reorder.

    Before the reorder, language ID ran first and could not identify a two-word string,
    so a length failure was reported as a language failure.
    """
    docs, stats = run([rec("s", "Too short.")])
    assert docs == []
    assert "too_short_chars" in stats.rejected
    assert "language" not in stats.rejected


def test_markup_is_stripped_before_the_length_gate():
    """A page that is mostly HTML is a short document. Filtering first would keep junk."""
    wrapped = "<div class='x'>" * 200 + PROSE + "</div>" * 200
    docs, _ = run([rec("h", wrapped)])
    assert len(docs) == 1
    assert "<div" not in docs[0].text


def test_exact_and_near_duplicates_are_both_dropped():
    docs, stats = run([
        rec("a", PROSE),
        rec("b", PROSE),                                   # exact
        rec("c", PROSE + " A final sentence was appended."),  # near
    ])
    assert len(docs) == 1
    assert stats.rejected["exact_duplicate"] == 1
    assert stats.rejected["near_duplicate"] == 1


def test_dedup_is_shared_across_sources():
    """FineWeb and FineWeb-Edu both derive from Common Crawl, so the same page can
    appear in both. Per-source dedup would let the pair through into different splits."""
    c = Cleaner()
    a = list(c.process([rec("x", PROSE)]))
    b = list(c.process([RawRecord(
        source_id="other", source_record_id="x", text=PROSE, license="unknown",
        domain="web", register="informational", acquired_at=datetime.now(timezone.utc))]))
    assert len(a) == 1 and len(b) == 0


def test_pii_is_redacted_and_flagged_on_the_record():
    docs, _ = run([rec("p", PROSE + " Write to clerk@harbour.example.org for the schedule.")])
    assert "clerk@harbour.example.org" not in docs[0].text
    assert "EMAIL" in docs[0].quality.pii_flags


def test_hash_matches_the_stored_text():
    """The hash is computed after redaction, so it must match what is actually stored,
    otherwise exact dedup and provenance disagree."""
    from forge.common.hashing import content_sha256

    docs, _ = run([rec("p", PROSE + " Write to clerk@harbour.example.org.")])
    assert docs[0].content_sha256 == content_sha256(docs[0].text)


def test_license_decides_redistributability():
    pd_docs, _ = run([rec("g", PROSE, license="public-domain-us")])
    web_docs, _ = run([rec("w", PROSE, license="ODC-By-1.0")])
    assert pd_docs[0].redistributable is True
    assert web_docs[0].redistributable is False


def test_no_group_leakage_across_a_batch():
    docs, _ = run([rec(str(i), PROSE.replace("harbour", f"harbour{i}")) for i in range(60)])
    check_no_group_leakage([(d.source_group_id, d.split) for d in docs])


def test_stats_account_for_every_input():
    inputs = [rec("a", PROSE), rec("b", "Too short."), rec("c", PROSE)]
    docs, stats = run(inputs)
    assert stats.seen == 3
    assert stats.kept + sum(stats.rejected.values()) == stats.seen


def test_publishable_artifact_is_not_inside_the_partition_tree(tmp_path):
    """Regression guard. metadata_only.parquet has no `text` column. If it sits inside
    the data lake, a glob read either raises a schema error or, worse, silently unions
    a text-less table into the training data."""
    import pyarrow.parquet as pq

    from forge.ingestion.writer import PARTITION_GLOB, write_metadata_only, write_parquet

    docs, _ = run([rec(str(i), PROSE.replace("harbour", f"port{i}")) for i in range(12)])
    lake = tmp_path / "lake"
    write_parquet(docs, lake)
    write_metadata_only(docs, tmp_path / "publishable" / "metadata_only.parquet")

    assert not list(lake.rglob("metadata_only.parquet"))
    files = sorted(lake.glob(PARTITION_GLOB))
    assert files
    schemas = {tuple(pq.read_schema(f).names) for f in files}
    assert len(schemas) == 1, "every partition file must share one schema"
    assert "text" in next(iter(schemas))


def test_cleaning_config_actually_reaches_the_cleaner():
    """Regression guard for a bug found on the first real ingestion run.

    The runner contained the line `policy.length.__dict__`, a no-op expression that reads
    an attribute and discards it. Every value under `cleaning:` was silently ignored and
    the LengthPolicy DEFAULTS ran instead: 200 / 50 / 20000 rather than the configured
    800 / 150 / 400.

    Nothing errored. The rejection counts looked healthy and the quota was met. Only the
    token-length distribution of the written corpus revealed it, and by then 60,000
    documents had been ingested.
    """
    from forge.common.config import load
    from forge.ingestion.run import policy_from_config

    p = policy_from_config(load("configs/data/human_minimal.yaml"))
    assert p.length.min_chars == 800
    assert p.length.min_tokens == 150
    assert p.length.max_tokens == 400


def test_defaults_survive_when_the_config_is_silent():
    from forge.cleaning.filters import LengthPolicy
    from forge.ingestion.run import policy_from_config

    p = policy_from_config({"cleaning": {}})
    assert p.length.max_tokens == LengthPolicy().max_tokens


def test_written_corpus_is_validated_against_the_config_not_the_plumbing():
    """Rejection accounting could not catch the bug: the counts were accurate for the
    policy that actually ran. What was missing was a check that the policy which ran was
    the policy that was requested."""
    from datetime import datetime, timezone

    import pytest

    from forge.common.config import load
    from forge.common.schemas import HumanDocument, Quality, Split
    from forge.ingestion.run import assert_corpus_matches_policy, policy_from_config

    policy = policy_from_config(load("configs/data/human_minimal.yaml"))
    doc = HumanDocument(
        doc_id="x", source_group_id="grp_x", text="t" * 10, source="fw",
        license="ODC-By-1.0", domain="web", text_register="informational",
        language_score=0.99, acquired_at=datetime.now(timezone.utc),
        processing_version="clean_v1", content_sha256="a" * 64, token_count=19639,
        quality=Quality(), split=Split.TRAIN,
    )
    with pytest.raises(RuntimeError, match="did not reach the cleaner"):
        assert_corpus_matches_policy([doc], policy)
