"""Forensic signals must describe the file, and must not over-read it.

WHY THIS FILE IS SHAPED THIS WAY. A forensic panel is read by someone who wants a yes or a
no, and the failure mode is not a crash. It is a plausible sentence that the evidence does
not support. Two of these tests exist purely to stop that:

  test_a_stripped_exif_says_it_proves_nothing
  test_standard_tables_do_not_imply_generation

Both assert on the CAVEAT text rather than the value, because the caveat is what stops a
reader concluding "no metadata, therefore AI" about a screenshot. Every image that has been
through a social platform, a chat app or a screenshot has no EXIF and library quantisation
tables. A tool that treats those as suspicious accuses the entire internet, and false
positives on ordinary human images are the one failure this whole project is organised
against.

The rest follow the discipline used throughout FORGE: test the ACCEPTING side of every rule,
not only the case where it fires.
"""

from __future__ import annotations

import io

import pytest

from forge.image.forensics import (
    HIGH,
    LOW,
    NOT_CHECKED,
    NOT_FOUND,
    PRESENT,
    analyze,
    c2pa_status,
    camera_consistency,
    error_level,
    exif,
    jpeg_tables,
    manipulation_summary,
    real_format,
    resample_evidence,
)

pytest.importorskip("PIL")
pytest.importorskip("numpy")


def _photo(width: int = 320, height: int = 240, fmt: str = "JPEG", quality: int = 92) -> bytes:
    """A textured image. NOT a flat colour.

    A blank image compresses to almost nothing, produces a zero compression residual and a
    degenerate spectrum, so every signal returns its trivial answer and the test proves
    nothing. Earlier fixtures in this project failed exactly this way three times.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:height, 0:width]
    base = (
        128
        + 60 * np.sin(x / 17.0)
        + 40 * np.cos(y / 11.0)
        + rng.normal(0, 12, size=(height, width))
    )
    stack = np.clip(np.dstack([base, base * 0.9 + 20, base * 1.05 - 10]), 0, 255)
    buffer = io.BytesIO()
    Image.fromarray(stack.astype("uint8"), "RGB").save(buffer, format=fmt, quality=quality)
    return buffer.getvalue()


def _with_exif() -> bytes:
    """A JPEG carrying camera-shaped EXIF, written through PIL's Exif container."""
    from PIL import Image

    with Image.open(io.BytesIO(_photo())) as img:
        raw = img.getexif()
        raw[271] = "Canon"
        raw[272] = "EOS R6"
        raw[306] = "2026:08:15 10:42:00"
        raw[36867] = "2026:08:15 10:42:00"
        raw[33437] = 4.0
        raw[33434] = 0.004
        raw[34855] = 200
        raw[37386] = 50.0
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=92, exif=raw)
    return buffer.getvalue()


# ------------------------------------------------------------------------------ container


def test_format_comes_from_the_bytes() -> None:
    finding = real_format(_photo(fmt="PNG"))
    assert finding.value == "PNG"
    assert finding.detail["width"] == 320


def test_an_unreadable_file_reports_that_rather_than_guessing() -> None:
    finding = real_format(b"this is not an image")
    assert finding.value is None
    assert finding.status == NOT_CHECKED


# ---------------------------------------------------------------------------------- EXIF


def test_exif_fields_are_read_not_invented() -> None:
    finding = exif(_with_exif())
    assert finding.status == PRESENT
    assert finding.value["camera_make"] == "Canon"
    assert finding.value["camera_model"] == "EOS R6"


def test_an_image_without_exif_reports_absence_not_a_default() -> None:
    """The regression that matters most: no placeholder camera, ever."""
    finding = exif(_photo())
    assert finding.value is None
    assert finding.status == NOT_FOUND


def test_a_stripped_exif_says_it_proves_nothing() -> None:
    """THE POINT OF THIS FILE.

    A reader seeing "EXIF: not found" will conclude something. The caveat is what decides
    whether that conclusion is correct. Screenshots and social-platform downloads have no
    EXIF, and treating them as suspicious means accusing ordinary people's photographs.
    """
    caveat = exif(_photo()).caveat.lower()
    assert "screenshot" in caveat
    assert "not evidence of generation" in caveat


def test_camera_consistency_recognises_a_full_camera_block() -> None:
    """The accepting side. A rule that only ever fires is not a rule."""
    assert camera_consistency(exif(_with_exif())).value == "consistent"


def test_camera_consistency_is_not_checked_without_exif() -> None:
    finding = camera_consistency(exif(_photo()))
    assert finding.status == NOT_CHECKED
    assert finding.value is None


# ---------------------------------------------------------------------------------- C2PA


def test_c2pa_absence_is_reported_as_normal() -> None:
    finding = c2pa_status(_photo())
    assert finding.status == NOT_FOUND
    assert "absence is" in finding.caveat


def test_c2pa_markers_are_detected_when_present() -> None:
    """The accepting side, using a file that genuinely carries the marker bytes."""
    doctored = _photo()[:200] + b"jumb" + _photo()[200:]
    assert c2pa_status(doctored).status == PRESENT


# --------------------------------------------------------------------------- JPEG tables


def test_quantisation_tables_are_read_from_a_jpeg() -> None:
    finding = jpeg_tables(_photo(quality=80))
    assert finding.status == PRESENT
    assert finding.value["estimated_quality"] is not None


def test_estimated_quality_tracks_the_quality_used() -> None:
    low = jpeg_tables(_photo(quality=40)).value["estimated_quality"]
    high = jpeg_tables(_photo(quality=95)).value["estimated_quality"]
    assert low < high


@pytest.mark.parametrize("quality", [30, 50, 70, 80, 88, 95])
def test_estimated_quality_is_accurate_not_merely_ordered(quality: int) -> None:
    """THE BUG THIS CAUGHT, and it was live for exactly one test run.

    libjpeg scales its base table by (200 - 2Q) for Q >= 50 and by 5000/Q below it. The
    first version of the inversion had those two branches the wrong way round, so every
    save above about q=78 came back as 100. A JPEG written at quality 80 was reported as
    lossless.

    The monotonic test above passed the whole time, because a broken monotonic function is
    still monotonic. Ordering is a weak property: it is satisfied by an infinite number of
    wrong functions, and the one thing it cannot detect is saturation. Asserting the actual
    value is what found it.
    """
    estimated = jpeg_tables(_photo(quality=quality)).value["estimated_quality"]
    assert abs(estimated - quality) <= 3, f"saved at q={quality}, estimated {estimated}"


def test_tables_are_not_checked_on_a_png() -> None:
    finding = jpeg_tables(_photo(fmt="PNG"))
    assert finding.status == NOT_CHECKED
    assert "not a JPEG" in finding.caveat


def test_standard_tables_do_not_imply_generation() -> None:
    """PIL writes scaled IJG tables, so this file reads as 'library defaults'.

    That is correct and it is also true of every re-saved photograph, which is why the
    caveat has to say so on the same line as the finding.
    """
    finding = jpeg_tables(_photo())
    assert finding.value["standard_tables"] is True
    assert "re-saved photograph" in finding.caveat


# ------------------------------------------------------------------------------ residual


def test_error_level_returns_a_measurement() -> None:
    finding = error_level(_photo())
    assert finding.value["mean_residual"] >= 0
    assert finding.status in {LOW, "medium", HIGH}


def test_error_level_caveats_that_any_reencode_triggers_it() -> None:
    assert "any re-encode" in error_level(_photo()).caveat


def test_a_corrupt_file_fails_soft_rather_than_returning_low() -> None:
    """Swallowing the error and returning a reassuring 'low' would be the project's
    recurring bug: a check that reports success without measuring anything."""
    finding = error_level(b"not an image at all")
    assert finding.status == NOT_CHECKED
    assert finding.value is None


def test_resample_reports_a_ratio_on_a_real_image() -> None:
    finding = resample_evidence(_photo())
    assert finding.value["peak_to_median"] > 0


def test_resample_is_skipped_on_a_tiny_image() -> None:
    finding = resample_evidence(_photo(width=32, height=32))
    assert finding.status == NOT_CHECKED


def test_resample_caveats_that_absence_proves_nothing() -> None:
    assert "does not mean" in resample_evidence(_photo()).caveat


# ------------------------------------------------------------------------------- summary


def test_summary_reports_the_strongest_signal_not_a_sum() -> None:
    """Deliberately not a score. These signals do not combine into a probability, and
    presenting one would dress a guess as a measurement."""
    findings = analyze(_photo())
    summary = manipulation_summary(findings)
    assert summary.value in {LOW, "medium", HIGH}
    assert "not a combined probability" in summary.caveat


def test_summary_is_not_checked_when_nothing_could_be_measured() -> None:
    assert manipulation_summary([]).status == NOT_CHECKED


def test_analyze_returns_every_signal() -> None:
    names = {f.name for f in analyze(_with_exif())}
    assert names == {
        "file_type",
        "exif",
        "camera_consistency",
        "c2pa",
        "ai_declaration",
        "jpeg_tables",
        "error_level",
        "resample",
        "noise_consistency",
        "colour",
        "manipulation_summary",
    }


def test_every_finding_carries_a_caveat_or_says_why_not() -> None:
    """A signal shown without its limit invites over-reading. Required, not optional."""
    for finding in analyze(_photo()):
        assert finding.caveat, f"{finding.name} has no caveat"
