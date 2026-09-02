"""Composites give the localisation head ground truth, and can hand it a shortcut instead.

THE TRAP. Pasting a generated patch into a photograph introduces a seam: a boundary between
two image statistics that is far more learnable than any generation artifact. A model
trained on such composites finds the seam, scores well on IoU, and locates nothing in a
real AI-edited image whose edit was blended.

THE DEFENCE that matters most is the control class. Human-region composites are built with
identical machinery from another photograph, so they carry the same seam and are labelled as
containing no AI. A model keying on the seam cannot separate the two classes, so
seam-learning stops paying. It is the same instrument as the negative control: build the
shortcut into both classes and it stops being a shortcut.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")
from PIL import Image  # noqa: E402

from forge.common.splits import assign_split  # noqa: E402
from forge.image.composites import (  # noqa: E402
    AREA_FRACTIONS,
    FEATHERS,
    SHAPES,
    build_composite,
    build_pairs,
    mask_area_fraction,
)
from forge.image.normalize import POLICY_V1, describe  # noqa: E402


def _image(seed: int, w: int = 700, h: int = 700) -> bytes:
    data = bytearray()
    for y in range(h):
        for x in range(w):
            grain = ((x * 7919 + y * 104729 + seed * 31) % 61) - 30
            data += bytes(
                (
                    max(0, min(255, (x * 200) // w + 30 + grain)),
                    max(0, min(255, (y * 180) // h + 40 + grain)),
                    max(0, min(255, 60 + seed * 7 + grain)),
                )
            )
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), bytes(data)).save(buf, format="PNG")
    return buf.getvalue()


PHOTOS = [(f"oi_{i:05d}", _image(i + 1)) for i in range(6)]
GENERATED = [_image(100 + i) for i in range(3)]


def test_composite_is_normalised_like_everything_else() -> None:
    image, _, _ = build_composite(PHOTOS[0][1], GENERATED[0], "oi_00000", "ai_region")
    got = describe(image)
    assert got["format"] == "JPEG"
    assert (got["width"], got["height"]) == (POLICY_V1.size, POLICY_V1.size)


def test_the_mask_marks_a_plausible_fraction() -> None:
    _, mask, record = build_composite(PHOTOS[0][1], GENERATED[0], "oi_00000", "ai_region")
    fraction = mask_area_fraction(mask)
    assert 0.02 < fraction < 0.6, fraction
    assert abs(fraction - record.area_fraction) < 0.2


def test_the_stored_mask_is_hard_not_feathered() -> None:
    """A soft mask makes IoU depend on an arbitrary threshold.

    The question being scored is which pixels came from the patch, which is a hard fact
    even when the blend that produced them was soft.
    """
    _, mask, _ = build_composite(PHOTOS[1][1], GENERATED[0], "oi_00001", "ai_region")
    with Image.open(io.BytesIO(mask)) as img:
        grey = img.convert("L")
        reader = getattr(grey, "get_flattened_data", None) or grey.getdata
        values = set(reader())
    assert values <= {0, 255}, f"mask contains intermediate values: {sorted(values)[:8]}"


def test_the_composite_actually_differs_from_its_base() -> None:
    """A composite identical to the original would teach nothing and score perfectly."""
    image, _, _ = build_composite(PHOTOS[0][1], GENERATED[0], "oi_00000", "ai_region")
    from forge.image.normalize import normalize_bytes

    assert image != normalize_bytes(PHOTOS[0][1])


# --- the control class, which is the point ----------------------------------------------


def test_pairs_are_balanced_between_ai_and_human_regions() -> None:
    """If the control class were smaller, the seam would still correlate with the label."""
    _, stats = build_pairs(PHOTOS, GENERATED)
    assert stats["ai_region"] == stats["human_region"] == len(PHOTOS)


def test_human_region_composites_are_labelled_zero() -> None:
    built, _ = build_pairs(PHOTOS, GENERATED)
    for _, _, record in built:
        assert record.label == (1 if record.kind == "ai_region" else 0)


def test_both_classes_are_built_by_the_same_machinery() -> None:
    """Same shape, feather and box for a given image, so only the pixels differ.

    If the control used different geometry, the geometry itself would become the signal.
    """
    built, _ = build_pairs(PHOTOS, GENERATED)
    by_image: dict[str, list] = {}
    for _, _, record in built:
        by_image.setdefault(record.source_image_id, []).append(record)
    for records in by_image.values():
        assert len({r.shape for r in records}) == 1
        assert len({r.feather for r in records}) == 1
        assert len({r.box for r in records}) == 1


def test_a_control_patch_never_comes_from_its_own_base() -> None:
    """Pasting an image into itself produces no seam and no change: a silent no-op."""
    from forge.image.normalize import normalize_bytes

    built, _ = build_pairs(PHOTOS, GENERATED)
    for image, _, record in built:
        if record.kind == "human_region":
            base = dict(PHOTOS)[record.source_image_id]
            assert image != normalize_bytes(base, allow_upscale=True)


# --- variation, so no single edge profile is the signal ----------------------------------


def test_shape_feather_and_area_all_vary_across_a_corpus() -> None:
    many = [(f"img_{i:05d}", _image(i)) for i in range(40)]
    built, _ = build_pairs(many, GENERATED)
    records = [record for _, _, record in built]
    assert len({r.shape for r in records}) == len(SHAPES)
    assert len({r.feather for r in records}) > 1
    assert len({r.area_fraction for r in records}) > 1
    assert set(r.feather for r in records) <= set(FEATHERS)
    assert set(r.area_fraction for r in records) <= set(AREA_FRACTIONS)


def test_hard_edges_are_included_deliberately() -> None:
    """Real edits are sometimes crude, so a zero-feather case must appear."""
    assert 0 in FEATHERS


# --- the invariants inherited from the text track -----------------------------------------


def test_a_composite_inherits_its_base_split() -> None:
    built, _ = build_pairs(PHOTOS, GENERATED)
    for _, _, record in built:
        assert record.split == assign_split(record.source_image_id).value
        assert record.source_group_id == record.source_image_id


def test_geometry_is_deterministic() -> None:
    first = build_composite(PHOTOS[2][1], GENERATED[0], "oi_00002", "ai_region")[2]
    second = build_composite(PHOTOS[2][1], GENERATED[0], "oi_00002", "ai_region")[2]
    assert first == second


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown composite kind"):
        build_composite(PHOTOS[0][1], GENERATED[0], "x", "sort_of_ai")


def test_building_without_generated_images_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to paste"):
        build_pairs(PHOTOS, [])


def test_building_from_a_single_photograph_is_refused() -> None:
    """The control patch has to come from somewhere other than the base."""
    with pytest.raises(ValueError, match="at least two"):
        build_pairs(PHOTOS[:1], GENERATED)
