"""The two-detector interface: every panel present, no panel invented.

The layout this serves has nine panels, and the temptation with a design that complete is
to fill the empty ones. Filling a verdict from metadata would fire hardest on screenshots
and re-saved photographs, which have no EXIF and library quantisation tables, so the tests
here assert that nothing is invented and that everything shown is genuinely computed.

READ THIS BEFORE ADDING A TEST HERE. This whole file was dead for most of the project:
`fastapi` was declared only in the `serve` extra, the dev venv did not install it, and
pytest skipped the module on an import error. Several tests in it then asserted the
pre-detector world ("no verdict, ever") and would have PASSED on any machine without model
weights while failing on a working build. So: assert the contract in both directions, and
branch on what the loaders report rather than pinning a state that happens to hold in CI.

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


def _post(data: bytes, name: str = "a.jpg", stability: bool = False):
    return client.post(
        "/v1/image/analyze",
        params={"stability": str(stability).lower()},
        files={"file": (name, data, "image/jpeg")},
    )


# ------------------------------------------------------------------------------- service


def test_health_reports_the_detectors_it_asked_rather_than_a_literal() -> None:
    """THE REGRESSION. /health returned hardcoded False for both detectors.

    It kept saying "detectors pending training" after both were built, and it could not
    have reported a real outage either, because no code path ever set those fields. The
    values must now come from the loaders, so this test asserts the SHAPE and the
    agreement with the loaders, not a fixed answer: whether a model is present depends on
    what is on disk, and a test that pins it to False is what let the lie survive.
    """
    from forge.image.detector import detector_state
    from forge.inference.scorer import available as text_arms

    body = client.get("/health").json()
    assert body["image_detector_loaded"] is bool(detector_state().get("available"))
    expected_text = bool([n for n, st in text_arms().items() if st == "ready"])
    assert body["text_detector_loaded"] is expected_text
    assert "pending training" not in body["mode"]
    for name, state in body["text_arms"].items():
        assert state, f"arm {name} reports neither ready nor a reason"


def test_the_page_serves_both_tabs() -> None:
    page = client.get("/").text
    assert 'data-tab="image"' in page and 'data-tab="text"' in page


# --------------------------------------------------------------------------- no invention


def test_the_image_verdict_matches_whether_a_detector_is_actually_loaded() -> None:
    """This asserted "no verdict, ever" and passed for the wrong reason.

    It passed because the machine running it had no detector weights, not because the code
    refuses to invent a verdict. The day the weights are cached locally it would have
    failed on a working build. The real contract has two halves and this checks both: with
    no verified detector the verdict is absent and a reason is given; with one, the verdict
    is one of the three the policy defines and carries a probability.
    """
    from forge.image.detector import detector_state

    state = detector_state()
    body = _post(_photo()).json()
    assessment = body["assessment"]
    if not state.get("available") or not state.get("polarity_verified"):
        assert assessment["available"] is False
        assert assessment["verdict"] is None
        assert assessment["confidence"] is None
        assert assessment["reason"], "a missing verdict must say why"
    else:
        assert assessment["available"] is True
        assert assessment["verdict"] in {"ai", "human", "uncertain"}
        assert 0.0 <= assessment["confidence"] <= 1.0


def test_no_attribution_heatmap_without_a_local_head() -> None:
    body = _post(_photo()).json()
    assert body["attribution"]["available"] is False
    assert body["attribution"]["heatmap"] is None
    reason = body["attribution"]["reason"].lower()
    # The word, not the meaning, is what this used to assert, so a rename broke it while
    # the behaviour was unchanged. Assert the two things that actually matter: the
    # segmentation is absent, and the reason does not deny the occlusion map that ships
    # alongside it under `attribution_map`.
    assert "localisation head" in reason
    assert "occlusion attribution" in reason


def test_the_visual_stream_is_never_a_zero_standing_in_for_a_missing_model() -> None:
    """A 0% bar reads as "the model says no". Unavailable reads as "there is no model".

    Same trap as the verdict test: pinning this to False passed only on a machine with no
    weights. What must hold either way is that `available` tracks the loader, and that an
    unavailable stream is not dressed up as a confident zero.
    """
    from forge.image.detector import detector_state

    loaded = bool(detector_state().get("available"))
    stream = {s["key"]: s for s in _post(_photo()).json()["evidence"]["streams"]}["visual_model"]
    assert stream["available"] is loaded
    if not loaded:
        assert stream.get("strength") in (None, 0)
        assert stream.get("direction") in (None, "neutral")


def test_the_text_endpoint_scores_with_both_arms_or_says_why_each_one_cannot() -> None:
    """This asserted a 503 and the word "fabricating", from before the arms were trained.

    It never ran, so it never failed when the endpoint started returning real scores. The
    contract now is: every arm appears either in `arms` with a score, or in `unavailable`
    with a reason. Neither list may silently swallow one.
    """
    from forge.inference.scorer import ARMS

    body = client.post("/v1/text/analyze", json={"text": "a sentence to analyse"}).json()
    accounted = {a["arm"] for a in body.get("arms", [])} | set(body.get("unavailable", {}))
    assert accounted == set(ARMS), f"arms unaccounted for: {set(ARMS) - accounted}"
    for arm in body.get("arms", []):
        # The payload key is `ai_probability`, not `probability`. Naming it from memory
        # rather than from the response is how a test ends up asserting a field that does
        # not exist, which is the same class of error as everything else fixed today.
        assert 0.0 <= arm["ai_probability"] <= 1.0
        assert 0.0 <= arm["max_window_probability"] <= 1.0
        assert arm["verdict"]
        assert arm["threshold"] is not None, "a verdict with no threshold is not a decision"
    for name, reason in (body.get("unavailable") or {}).items():
        assert reason.strip(), f"arm {name} is unavailable with no stated reason"


# ------------------------------------------------------------------- what IS computed


def test_every_panel_is_populated() -> None:
    """Stability is requested explicitly, because the pass is opt-in on the endpoint.

    This used to post without the flag and expect the panel filled, from when the pass ran
    on every upload. Posting without it and demanding content asserts the opposite of the
    documented behaviour, which is that absence is reported as absence.
    """
    body = _post(_photo(exif=True), stability=True).json()
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
    body = _post(_photo(exif=True), stability=True).json()
    assert body["stability_available"] is True
    lost = [r for r in body["stability"] if r["value"] and "lost EXIF" in r["value"]]
    assert lost, "a re-encode should destroy the EXIF block"


def test_conflict_is_low_when_nothing_disagrees() -> None:
    assert _post(_photo()).json()["evidence"]["conflict"] == "low"


def test_the_report_lists_what_it_cannot_conclude() -> None:
    lines = " ".join(_post(_photo()).json()["cannot_conclude"]).lower()
    assert "whether this image was generated" in lines
    # THE REGRESSION. This section printed "the trained detector, which does not exist yet"
    # directly under a verdict the detector had just produced. The honest limitation depends
    # on whether one ran, so the sentence has to depend on it too.
    assert "does not exist yet" not in lines
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


def test_the_app_exposes_every_route_the_page_calls() -> None:
    """THE REGRESSION. `app = FastAPI(...)` was deleted by an edit that moved code out.

    Extracting the image analysis into forge.image.analysis meant cutting a block out of
    this module, and the cut took the application object with it. The module then failed to
    import at all, which pytest reports as a COLLECTION ERROR rather than a failure, and a
    collection error is the easiest thing in a test run to scroll past.

    Asserting the route table rather than just the import also catches the subtler version:
    a route that survives the file but no longer answers, because its decorator went with
    the block that moved.
    """
    paths = {getattr(route, "path", None) for route in app.routes}
    for required in ("/", "/health", "/v1/image/analyze", "/v1/text/analyze",
                     "/v1/image/detector", "/v1/text/arms", "/v1/results"):
        assert required in paths, f"{required} is no longer served"
