"""A manifest describes a corpus without redistributing it.

Open Images lists its images as CC BY 2.0 while its maintainers explicitly disclaim any
warranty about the license status of any individual image. FORGE therefore records the
license as a claim carried from the source and never publishes pixels. What ships is ids,
URLs, license, attribution, a perceptual hash and a split.
"""

from __future__ import annotations

import json

import pytest

from forge.image.manifest import (
    ImageRecord,
    assert_split_disjoint,
    read_manifest,
    split_counts,
    write_manifest,
)


def _record(i: int, **kw) -> ImageRecord:
    base = dict(
        image_id=f"oi_{i:06d}",
        source="open_images_v7",
        source_url=f"https://example.invalid/{i}.jpg",
        license="CC BY 2.0 (as recorded upstream)",
        attribution=f"Photographer {i}",
        phash=f"{i:016x}",
        width=1600,
        height=1200,
        original_format="JPEG",
    )
    base.update(kw)
    return ImageRecord(**base).with_split()


RECORDS = [_record(i) for i in range(600)]


def test_every_record_gets_a_split() -> None:
    assert all(r.split for r in RECORDS)


def test_splits_are_roughly_the_configured_ratios() -> None:
    counts = split_counts(RECORDS)
    assert counts["train"] > counts["val"] > 0
    assert counts["train"] > counts["test"] > 0
    assert abs(counts["train"] / len(RECORDS) - 0.80) < 0.06


def test_split_assignment_is_deterministic() -> None:
    """Two runs must build the same corpus, or nothing downstream is reproducible."""
    assert [r.split for r in RECORDS] == [_record(i).split for i in range(600)]


def test_writing_and_reading_round_trips(tmp_path) -> None:
    out = tmp_path / "manifest.jsonl"
    summary = write_manifest(RECORDS, out)
    assert summary["images"] == 600
    assert read_manifest(out) == sorted(RECORDS, key=lambda r: r.image_id)


def test_manifest_is_byte_stable(tmp_path) -> None:
    """A diff should mean the corpus changed, not that the filesystem reordered it."""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_manifest(RECORDS, a)
    write_manifest(list(reversed(RECORDS)), b)
    assert a.read_bytes() == b.read_bytes()


def test_manifest_contains_no_pixels(tmp_path) -> None:
    """The rule the whole module exists to enforce."""
    out = tmp_path / "m.jsonl"
    write_manifest(RECORDS[:5], out)
    for line in out.read_text().splitlines():
        row = json.loads(line)
        assert set(row) == {
            "image_id", "source", "source_url", "license", "attribution", "phash",
            "width", "height", "original_format", "norm_policy", "split", "notes",
        }


def test_a_field_carrying_image_bytes_is_refused(tmp_path) -> None:
    """Guards against a future 'thumbnail' field quietly becoming redistribution."""
    smuggled = _record(1, notes={"ok": "fine"})
    object.__setattr__(smuggled, "attribution", "A" * 5000)
    with pytest.raises(ValueError, match="embedded image data"):
        write_manifest([smuggled], tmp_path / "m.jsonl")


def test_an_unsplit_record_is_refused(tmp_path) -> None:
    """An unsplit manifest cannot be checked for leakage, so it must not be written."""
    unsplit = ImageRecord(
        image_id="x", source="s", source_url="u", license="l", attribution="a",
        phash="0" * 16, width=10, height=10, original_format="JPEG",
    )
    with pytest.raises(ValueError, match="no split"):
        write_manifest([unsplit], tmp_path / "m.jsonl")


def test_duplicate_ids_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate image_id"):
        write_manifest([_record(1), _record(1)], tmp_path / "m.jsonl")


def test_split_disjointness_is_checked_not_trusted() -> None:
    good = RECORDS[:50]
    assert_split_disjoint(good)
    bad = list(good) + [ImageRecord(**{**good[0].__dict__, "split": "test"})]
    if good[0].split != "test":
        with pytest.raises(RuntimeError, match="inflated metrics"):
            assert_split_disjoint(bad)


def test_license_is_recorded_as_a_claim_not_a_verdict() -> None:
    """The string must carry its provenance, because upstream disclaims it."""
    assert "as recorded upstream" in RECORDS[0].license
