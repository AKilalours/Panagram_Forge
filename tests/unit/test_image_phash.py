"""Near-duplicate detection must survive re-encoding and catch real duplicates.

WHY. If a photograph appears twice and one copy is mirrored into the AI class, the detector
sees near-identical content on both sides of the label and learns whichever incidental
difference separates them. Duplicates that straddle a split inflate every metric.

Exact hashing cannot find these: two JPEG encodings of one photograph share no bytes.
Dhash compares each pixel to its right-hand neighbour, so it encodes gradients and survives
the requantisation, brightness shifts and resizing an image collects on its way through a
hosting pipeline.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")
from PIL import Image, ImageEnhance  # noqa: E402

from forge.image.phash import (  # noqa: E402
    DuplicateIndex,
    dhash,
    distance,
    is_near_duplicate,
    to_hex,
)


def _scene(seed: int, w: int = 600, h: int = 600) -> Image.Image:
    """A deterministic image with structure. Flat colour would hash identically for all."""
    img = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            img.putpixel((x, y), ((x * seed) % 256, (y * 7 + seed) % 256, ((x ^ y) + seed) % 256))
    return img


def _bytes(img: Image.Image, fmt: str = "JPEG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


SCENE = _scene(3)
OTHER = _scene(29)


def test_identical_bytes_hash_identically() -> None:
    raw = _bytes(SCENE)
    assert dhash(raw) == dhash(raw)


def test_recompression_does_not_change_the_hash_much() -> None:
    """The case exact hashing cannot handle: same photo, different JPEG quality."""
    a = dhash(_bytes(SCENE, quality=95))
    b = dhash(_bytes(SCENE, quality=45))
    assert is_near_duplicate(a, b), f"distance was {distance(a, b)}"


def test_format_change_does_not_change_the_hash_much() -> None:
    a = dhash(_bytes(SCENE, "PNG"))
    b = dhash(_bytes(SCENE, "JPEG", quality=80))
    assert is_near_duplicate(a, b), f"distance was {distance(a, b)}"


def test_moderate_resize_does_not_change_the_hash_much() -> None:
    a = dhash(_bytes(SCENE))
    b = dhash(_bytes(SCENE.resize((300, 300), Image.LANCZOS)))
    assert is_near_duplicate(a, b), f"distance was {distance(a, b)}"


def test_brightness_change_does_not_change_the_hash_much() -> None:
    """Why dhash and not average hash: gradients survive, absolute levels do not matter."""
    a = dhash(_bytes(SCENE))
    b = dhash(_bytes(ImageEnhance.Brightness(SCENE).enhance(1.25)))
    assert is_near_duplicate(a, b), f"distance was {distance(a, b)}"


def test_different_images_are_not_near_duplicates() -> None:
    """A hash that called everything a duplicate would pass every test above."""
    assert not is_near_duplicate(dhash(_bytes(SCENE)), dhash(_bytes(OTHER)))


def test_hex_is_fixed_width() -> None:
    """Manifests must sort and compare predictably."""
    assert len({len(to_hex(dhash(_bytes(_scene(i))))) for i in range(1, 6)}) == 1


def test_distance_bounds() -> None:
    assert distance(0, 0) == 0
    assert distance(0, (1 << 64) - 1) == 64


def test_index_keeps_the_first_and_reports_the_original() -> None:
    index = DuplicateIndex()
    assert index.add(dhash(_bytes(SCENE, quality=95)), "first") is None
    assert index.add(dhash(_bytes(SCENE, quality=40)), "second") == "first"
    assert len(index) == 1


def test_index_keeps_distinct_images() -> None:
    index = DuplicateIndex()
    index.add(dhash(_bytes(SCENE)), "a")
    assert index.add(dhash(_bytes(OTHER)), "b") is None
    assert len(index) == 2
