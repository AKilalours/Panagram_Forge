"""Normalisation must erase every trace of how an image was encoded.

WHY THESE TESTS EXIST. A human/AI image detector can score near-perfectly by learning that
photographs are JPEG, 4:3, EXIF-bearing and resized by a host, while diffusion output is
lossless PNG, 1:1, metadata-free and exactly 1024x1024. That model has learned two encoding
pipelines, not generation. Accuracy, AUROC and every headline metric look excellent while
it happens.

Each test below names one channel through which the source could leak, and asserts that
normalisation closes it.
"""

from __future__ import annotations

import io

import pytest

PIL = pytest.importorskip("PIL", reason="pillow is in the [image] extra")
from PIL import Image  # noqa: E402

from forge.image.normalize import (  # noqa: E402
    POLICY_V1,
    ImageTooSmallError,
    NormalizationPolicy,
    describe,
    normalize_bytes,
)


def _image(w: int, h: int, fmt: str = "JPEG", colour=(120, 90, 60), **save_kw) -> bytes:
    """A plausible image with structure, not flat colour, so JPEG has something to do."""
    img = Image.new("RGB", (w, h), colour)
    for x in range(0, w, 7):
        for y in range(0, h, 11):
            img.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kw)
    return buf.getvalue()


PHOTO = _image(1600, 1200, "JPEG", quality=72)          # 4:3, lossy, like a hosted photo
GENERATED = _image(1024, 1024, "PNG")                    # 1:1, lossless, like diffusion out


def test_output_is_always_jpeg() -> None:
    for raw in (PHOTO, GENERATED):
        assert describe(normalize_bytes(raw))["format"] == "JPEG"


def test_output_is_always_the_policy_size() -> None:
    """Aspect ratio is a source signature: photographs are 4:3, diffusion output is 1:1."""
    for raw in (PHOTO, GENERATED):
        got = describe(normalize_bytes(raw))
        assert (got["width"], got["height"]) == (POLICY_V1.size, POLICY_V1.size)


def test_photo_and_generated_are_indistinguishable_by_container() -> None:
    """The regression this whole module exists for.

    After normalisation, format, dimensions, mode and metadata presence must carry no
    information about which population an image came from.
    """
    a = describe(normalize_bytes(PHOTO))
    b = describe(normalize_bytes(GENERATED))
    assert a == b, f"container fields still differ between populations: {a} vs {b}"


def test_metadata_is_stripped() -> None:
    img = Image.new("RGB", (900, 900), (10, 20, 30))
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[271] = "ACME Camera Co"        # Make
    img.save(buf, format="JPEG", exif=exif)
    assert describe(normalize_bytes(buf.getvalue()))["has_exif"] is False


def test_exif_rotation_is_applied_before_it_is_discarded() -> None:
    """Order matters, and getting it wrong corrupts one class only.

    Phone photographs are frequently stored sideways with an orientation flag. Dropping the
    flag before applying it leaves the centre crop taking the wrong region of every such
    image, systematically and silently, and only for the human population.
    """
    tall = Image.new("RGB", (400, 800), (0, 0, 0))
    tall.paste(Image.new("RGB", (400, 100), (255, 0, 0)), (0, 0))   # red band at the top
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6      # rotate 90 CW on display
    tall.save(buf, format="JPEG", exif=exif, quality=95)

    out = Image.open(io.BytesIO(normalize_bytes(buf.getvalue(), NormalizationPolicy(size=200))))
    # After transposition the red band is on one SIDE, so the left and right halves differ.
    left = out.crop((0, 0, 20, 200)).resize((1, 1)).getpixel((0, 0))
    right = out.crop((180, 0, 200, 200)).resize((1, 1)).getpixel((0, 0))
    assert left != right, "orientation was discarded before it was applied"


def test_alpha_and_palette_are_flattened() -> None:
    """Channel count is a source signature: generated PNGs carry alpha, photographs do not."""
    rgba = Image.new("RGBA", (800, 800), (10, 200, 30, 128))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    assert describe(normalize_bytes(buf.getvalue()))["mode"] == "RGB"


def test_upscaling_is_refused_rather_than_performed() -> None:
    """Interpolation fabricates high-frequency detail, which is what the model looks at."""
    small = _image(300, 300, "JPEG")
    with pytest.raises(ImageTooSmallError):
        normalize_bytes(small, NormalizationPolicy(size=512))


def test_normalisation_is_deterministic() -> None:
    assert normalize_bytes(PHOTO) == normalize_bytes(PHOTO)


def test_renormalising_does_not_change_the_container() -> None:
    """A second pass must not change shape or format, or stages would disagree."""
    once = normalize_bytes(PHOTO)
    twice = normalize_bytes(once)
    assert describe(once) == describe(twice)


def test_policy_is_versioned() -> None:
    """Changing the policy changes what every model has ever seen; it needs a version."""
    assert POLICY_V1.version
    assert NormalizationPolicy(size=256).size == 256


@pytest.mark.parametrize("bad", [{"size": 8}, {"jpeg_quality": 0}, {"jpeg_quality": 101}])
def test_nonsense_policies_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        NormalizationPolicy(**bad)
