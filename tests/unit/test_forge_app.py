"""The two-detector interface: every panel present, no panel invented.

The layout this serves has nine panels, and the temptation with a design that complete is
to fill the empty ones. Two of them need trained weights: the overall assessment and the
attribution heatmap. Filling those from metadata would fire hardest on screenshots and
re-saved photographs, which have no EXIF and library quantisation tables, so the tests here
assert those two stay empty and that everything else is genuinely computed.

`test_a_declared_ai_image_is_detected_without_any_model` is the interesting one. Metadata
declarations are the only model-free signal that can positively indicate generation, and
the pair of tests around it fixes the asymmetry in place: present is strong, absent is
silence.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.forge_app import MAX_IMAGE_BYTES, app  # noqa: E402

client = TestClient(app)


def _photo(fmt: str = "JPEG", exif: bool = False, text_chunk: dict | None = None) -> bytes:
    import numpy as np
    from PIL import Image, PngImagePlugin

    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:240, 0:320]
    base = 125 + 48 * np.sin(x / 15.0) + 30 * np.cos(y / 9.0) + rng.normal(0, 9, (240, 320))
    img = Image.fromarray(
        np.clip(np.dstack([base * 1.02, base * 0.96, base * 0.9]), 0, 255).astype("uint8"), "RGB"
    )
    buffer = io.BytesIO()
    if fmt == "PNG" and text_chunk:
        info = PngImagePlugin.PngInfo()
        for key, value in text_chunk.items():
            info.add_text(key, value)
        img.save(buffer, format="PNG", pnginfo=info)
    elif exif:
        raw = img.getexif()
        raw[271], raw[272] = "Canon", "EOS R6"
        raw[36867] = "2026:08:15 10:42:00"
        raw[33437], raw[33434], raw[34855], raw[37386] = 4.0, 0.004, 200, 50.0
        img.save(buffer, format="JPEG", quality=90, exif=raw)
    else:
        img.save(buffer, format=fmt, quality=90)
    return buffer.getvalue()


def _post(data: bytes, name: str = "a.jpg"):
    return client.post("/v1/image/analyze", files={"file": (name, data, "image/jpeg")})


# ------------------------------------------------------------------------------- service


def test_health_declares_both_detectors_absent() -> None:
    body = client.get("/health").json()
    assert body["image_detector_loaded"] is False
    assert body["text_detector_loaded"] is False


def test_the_page_serves_both_tabs() -> None:
    page = client.get("/").text
    assert 'data-tab="image"' in page and 'data-tab="text"' in page


# --------------------------------------------------------------------------- no invention


def test_no_image_verdict_without_a_detector() -> None:
    body = _post(_photo()).json()
    assert body["assessment"]["available"] is False
    assert body["assessment"]["verdict"] is None
    assert body["assessment"]["confidence"] is None


def test_no_attribution_heatmap_without_a_local_head() -> None:
    body = _post(_photo()).json()
    assert body["attribution"]["available"] is False
    assert body["attribution"]["heatmap"] is None
    assert "untrained" in body["attribution"]["reason"]


def test_the_visual_stream_is_marked_unavailable_rather_than_zero() -> None:
    """A 0% bar reads as 'the model says no'. Unavailable reads as 'there is no model'."""
    streams = {s["key"]: s for s in _post(_photo()).json()["evidence"]["streams"]}
    assert streams["visual_model"]["available"] is False


def test_the_text_tab_surfaces_the_api_reason_rather_than_a_score() -> None:
    body = client.post("/v1/text/analyze", json={"text": "a sentence to analyse"}).json()
    assert body["available"] is False
    assert body["status"] == 503
    assert "fabricating" in body["reason"]


# ------------------------------------------------------------------- what IS computed


def test_every_panel_is_populated() -> None:
    body = _post(_photo(exif=True)).json()
    for panel in ("authenticity", "manipulation", "provenance", "stability"):
        assert body[panel], f"{panel} is empty"
    assert len(body["evidence"]["streams"]) == 5


def test_camera_fields_come_from_the_file() -> None:
    rows = {r["label"]: r["value"] for r in _post(_photo(exif=True)).json()["authenticity"]}
    assert rows["Camera"] == "Canon EOS R6"
    assert rows["Capture date"] == "2026:08:15 10:42:00"


def test_camera_fields_are_absent_when_the_file_has_none() -> None:
    """No placeholder camera, ever. The reference design shows one; ours shows nothing."""
    rows = {r["label"]: r["value"] for r in _post(_photo()).json()["authenticity"]}
    assert rows["Camera"] is None
    assert rows["Capture date"] is None


def test_a_declared_ai_image_is_detected_without_any_model() -> None:
    """THE MODEL-FREE WIN.

    Stable Diffusion front-ends write the prompt, sampler and model hash into a PNG text
    chunk. Reading it identifies generated content with no weights, no GPU and no inference.
    """
    chunk = {
        "parameters": "a cat on a windowsill\\nNegative prompt: blurry\\n"
        "Steps: 30, Sampler: DPM++ 2M, CFG scale: 7, Model hash: a1b2c3"
    }
    body = client.post(
        "/v1/image/analyze",
        files={"file": ("gen.png", _photo("PNG", text_chunk=chunk), "image/png")},
    ).json()
    streams = {s["key"]: s for s in body["evidence"]["streams"]}
    assert streams["self_declaration"]["direction"] == "toward_ai"
    assert streams["self_declaration"]["strength"] >= 90


def test_an_undeclared_image_is_not_called_human() -> None:
    """The other half of the asymmetry, and the more important half.

    Absence of a declaration must move nothing. Most generators write none, and every
    editor strips them, so treating silence as evidence of human authorship would accuse
    and exonerate at random.
    """
    streams = {s["key"]: s for s in _post(_photo()).json()["evidence"]["streams"]}
    declaration = streams["self_declaration"]
    assert declaration["direction"] == "neutral"
    assert declaration["strength"] == 0
    assert "means nothing" in declaration["note"]


def test_stability_shows_which_transforms_destroy_which_signals() -> None:
    """Real robustness data, computed without a model: EXIF does not survive a re-save."""
    body = _post(_photo(exif=True)).json()
    assert body["stability_available"] is True
    lost = [r for r in body["stability"] if r["value"] and "lost EXIF" in r["value"]]
    assert lost, "a re-encode should destroy the EXIF block"


def test_conflict_is_low_when_nothing_disagrees() -> None:
    assert _post(_photo()).json()["evidence"]["conflict"] == "low"


def test_the_report_lists_what_it_cannot_conclude() -> None:
    lines = " ".join(_post(_photo()).json()["cannot_conclude"]).lower()
    assert "whether this image was generated" in lines
    assert "screenshots" in lines


# -------------------------------------------------------------------------------- refusal


def test_an_empty_upload_is_refused() -> None:
    assert _post(b"").status_code == 400


def test_a_non_image_is_refused() -> None:
    assert _post(b"#!/bin/sh\necho hi\n", name="s.sh").status_code == 415


def test_an_oversized_upload_is_refused() -> None:
    assert _post(b"\xff\xd8" + b"\x00" * (MAX_IMAGE_BYTES + 1)).status_code == 413


def test_a_png_is_accepted() -> None:
    """The accepting side of the format rule; refusing everything would pass a refusal test."""
    response = client.post(
        "/v1/image/analyze", files={"file": ("a.png", _photo("PNG"), "image/png")}
    )
    assert response.status_code == 200
