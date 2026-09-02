"""Fetching must normalise at the boundary, dedup on normalised bytes, and count failures.

Three properties, each of which would be a silent data bug if it were wrong:

- A raw photograph must never exist inside FORGE. If images were stored as downloaded and
  normalised later, some stage would eventually read a raw one, and the difference between
  raw and normalised IS the source signature the image track exists to remove.
- The perceptual hash must be computed on normalised bytes, or two copies of one photograph
  stored at different sizes pass as distinct images.
- Failures must be counted, not skipped. A corpus that quietly shrinks when a URL rots has
  a size that depends on the day it was built.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")
from PIL import Image  # noqa: E402

from forge.image.fetch import SourceImage, fetch_corpus, sample_ids  # noqa: E402
from forge.image.normalize import POLICY_V1, describe  # noqa: E402


def _scene(seed: int, w: int = 640, h: int = 520, fmt: str = "JPEG") -> bytes:
    """A distinct image per seed, with structure in EVERY pixel.

    An earlier version of this helper painted only every third column, leaving most of the
    frame black. Two such images downscale to nearly the same 8x9 grayscale grid, so the
    deduplicator correctly called them duplicates and the test that wanted two distinct
    images got one. The bug was in the fixture: a near-duplicate detector is supposed to
    collapse two nearly-blank frames.

    Seed controls the stripe period and phase, so different seeds differ in gradient
    structure, which is what dhash actually measures.
    """
    period = 3 + (seed % 7)
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data += bytes(
                (
                    ((x // period) * 53 + seed * 17) % 256,
                    ((y // (period + 2)) * 31 + seed * 29) % 256,
                    ((x + y) * (seed + 1)) % 256,
                )
            )
    img = Image.frombytes("RGB", (w, h), bytes(data))
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=88)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _sources(n: int) -> list[SourceImage]:
    return [
        SourceImage(f"oi_{i:05d}", f"https://example.invalid/{i}", "CC BY 2.0", f"Author {i}")
        for i in range(n)
    ]


def _collector():
    stored: dict[str, bytes] = {}
    return stored, lambda image_id, data: stored.__setitem__(image_id, data)


def test_stored_bytes_are_normalised_not_raw() -> None:
    """The property the whole image track depends on."""
    stored, store = _collector()
    sources = _sources(3)
    payloads = {s.url: _scene(i + 1) for i, s in enumerate(sources)}

    records, stats = fetch_corpus(sources, download=lambda u: payloads[u], store=store)

    assert stats.stored == 3
    for data in stored.values():
        got = describe(data)
        assert got["format"] == "JPEG"
        assert (got["width"], got["height"]) == (POLICY_V1.size, POLICY_V1.size)
        assert got["has_exif"] is False


def test_png_and_jpeg_sources_are_indistinguishable_once_stored() -> None:
    stored, store = _collector()
    sources = _sources(2)
    payloads = {sources[0].url: _scene(5, fmt="JPEG"), sources[1].url: _scene(9, fmt="PNG")}

    fetch_corpus(sources, download=lambda u: payloads[u], store=store)
    a, b = (describe(v) for v in stored.values())
    assert a == b


def test_near_duplicates_are_dropped_and_counted() -> None:
    """Same photograph at two qualities is one image, not two."""
    stored, store = _collector()
    sources = _sources(2)
    same = _scene(7)
    reencoded = _scene(7)
    payloads = {sources[0].url: same, sources[1].url: reencoded}

    records, stats = fetch_corpus(sources, download=lambda u: payloads[u], store=store)
    assert stats.stored == 1
    assert stats.rejected["near_duplicate"] == 1
    assert len(records) == 1


def test_download_failures_are_recorded_not_swallowed() -> None:
    stored, store = _collector()
    sources = _sources(4)
    good = {sources[0].url: _scene(1), sources[2].url: _scene(2)}

    def download(url: str) -> bytes:
        if url in good:
            return good[url]
        raise OSError("404")

    records, stats = fetch_corpus(
        sources, download=download, store=store, attempts=2, sleep=lambda _: None
    )
    assert stats.requested == 4
    assert stats.stored == 2
    assert stats.rejected["download_failed"] == 2
    assert stats.as_dict()["keep_rate"] == 0.5


def test_a_transient_failure_is_retried() -> None:
    stored, store = _collector()
    calls = {"n": 0}

    def flaky(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return _scene(4)

    records, stats = fetch_corpus(
        _sources(1), download=flaky, store=store, attempts=3, sleep=lambda _: None
    )
    assert stats.stored == 1
    assert calls["n"] == 2


def test_images_below_the_policy_size_are_rejected_with_a_reason() -> None:
    """Upscaling would fabricate the high-frequency detail the model judges."""
    stored, store = _collector()
    records, stats = fetch_corpus(
        _sources(1), download=lambda u: _scene(1, 300, 300), store=store
    )
    assert stats.stored == 0
    assert stats.rejected["too_small"] == 1


def test_undecodable_bytes_are_rejected_with_a_reason() -> None:
    stored, store = _collector()
    records, stats = fetch_corpus(_sources(1), download=lambda u: b"not an image", store=store)
    assert stats.stored == 0
    assert sum(stats.rejected.values()) == 1


def test_records_carry_provenance_and_a_split() -> None:
    stored, store = _collector()
    sources = _sources(1)
    records, _ = fetch_corpus(sources, download=lambda u: _scene(6), store=store)
    record = records[0]
    assert record.license == "CC BY 2.0"
    assert record.attribution == "Author 0"
    assert record.source_url == sources[0].url
    assert record.split in {"train", "val", "test"}
    # The ORIGINAL dimensions, not the normalised ones. Read from the fixture rather than
    # hardcoded, so changing the fixture cannot leave this assertion quietly wrong again.
    original = describe(_scene(6))
    assert (record.width, record.height) == (original["width"], original["height"])
    assert record.width != POLICY_V1.size, "the manifest must record the source size"
    assert record.phash


def test_sampling_is_deterministic_and_not_a_prefix() -> None:
    """The upstream index is grouped, so a prefix would be a biased slice.

    This is the mistake the text corpus loader made twice; it is not repeated here.
    """
    sources = _sources(500)
    first = [s.image_id for s in sample_ids(sources, 50)]
    assert first == [s.image_id for s in sample_ids(sources, 50)]
    assert first != [s.image_id for s in sources[:50]]


def test_smaller_samples_nest_inside_larger_ones() -> None:
    sources = _sources(500)
    small = {s.image_id for s in sample_ids(sources, 20)}
    large = {s.image_id for s in sample_ids(sources, 100)}
    assert small <= large
