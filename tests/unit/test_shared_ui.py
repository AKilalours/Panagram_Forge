"""One stylesheet and one set of result cards, shared by both shells.

WHY THIS FILE EXISTS. There are now two pages over the same detectors: the FastAPI page in
api/forge_app.py and the Streamlit page in streamlit_app.py, which is what actually deploys
because free Docker hosting went away. Two pages rendering one payload is precisely the
shape of every wrong answer this project has shipped: a second copy that nobody updates
when the first is corrected. These tests hold the two together at the seams that matter.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from forge.ui.render import banner, image_result, text_result  # noqa: E402
from forge.ui.theme import CSS  # noqa: E402


def _payload(**overrides) -> dict:
    base = {
        "assessment": {"available": True, "verdict": "human", "confidence": 0.009,
                       "reason": "", "detail": ""},
        "evidence": {"conflict": "low", "streams": [
            {"key": "visual_model", "label": "Visual detector", "strength": 98,
             "state": "detected", "direction": "toward_capture",
             "summary": "P(AI) = 0.94%", "available": True},
            {"key": "camera", "label": "Camera metadata", "strength": 50, "state": "partial",
             "direction": "neutral", "summary": "4 fields, partial", "available": True},
        ]},
        "authenticity": [{"label": "Camera", "value": "Canon EOS R6", "status": "present"}],
        "provenance": [{"label": "C2PA signature", "value": None, "status": "not_found"}],
        "manipulation": [],
        "cannot_conclude": ["Whether this image was generated, with confidence."],
        "preview": None, "filename": "a.jpg", "size_bytes": 3050000,
    }
    base.update(overrides)
    return base


def test_the_stylesheet_is_not_duplicated_in_either_page():
    """A copied stylesheet is a stylesheet that diverges. Both pages import this one."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    fastapi_page = (root / "api" / "forge_app.py").read_text(encoding="utf-8")
    streamlit_page = (root / "streamlit_app.py").read_text(encoding="utf-8")

    assert "%%FORGE_CSS%%" in fastapi_page, "the FastAPI page no longer splices in the shared CSS"
    # Match the SELECTOR, not the word: "d.assessment" in the page's JavaScript contains
    # ".assess" and made the first version of this test fail on correct code.
    for page, name in ((fastapi_page, "FastAPI"), (streamlit_page, "Streamlit")):
        assert ".assess{" not in page, f"a second copy of the card styles is inline in {name}"
        assert ".gauge{" not in page, f"a second copy of the gauge styles is inline in {name}"
    assert ".assess{" in CSS and ".stream{" in CSS and ".gauge{" in CSS


def test_the_verdict_word_is_never_human():
    """NO AI DETECTED, never HUMAN, in both shells.

    A detector cannot establish that a person made something; it reports whether the input
    resembles the distribution it was trained on. This is the single most load-bearing
    wording choice in the project and it must survive a second renderer.
    """
    rendered = banner(available=True, verdict="human", probability=0.01)
    headline = rendered[rendered.index("<h2>"):rendered.index("</h2>")]
    assert headline.strip("<h2>") == "NO AI DETECTED"
    # The gauge's end labels legitimately read human / uncertain / AI, so the check is on
    # the HEADLINE only. The first version of this test searched the whole card and failed
    # on correct output.
    assert "HUMAN" not in headline.upper().replace("NO AI DETECTED", "")


def test_an_unavailable_verdict_shows_no_score_and_no_gauge():
    """No detector means no number, and no gauge implying one exists."""
    rendered = banner(available=False, verdict=None, probability=None, reason="Detector not configured.")
    assert "No verdict" in rendered
    assert "not available" in rendered
    assert "gauge" not in rendered, "a gauge with no score invites the reader to read a position"


def test_evidence_streams_show_the_word_not_the_number():
    """`strength` is how much a stream has to say, not a probability.

    Rendered as "50%" beside a camera-metadata row it reads as "50% likely to be a
    photograph", which is not what it means and is not recoverable from a caption. The
    FastAPI page shows `state`; so must this one.
    """
    html = image_result(_payload())
    assert "partial" in html and "detected" in html
    # The bar's WIDTH is still the strength, which is what a bar is for. What must not
    # appear is the number as text the reader can mistake for a probability.
    assert ">50%<" not in html and "> 50%" not in html
    assert "width:50%" in html, "the bar should still be sized by the strength"


def test_a_stream_without_a_state_raises_rather_than_rendering_an_empty_pill():
    """Guessing a payload key is how a pill ships blank. This one was `strength_label`."""
    payload = _payload()
    del payload["evidence"]["streams"][0]["state"]
    with pytest.raises(KeyError):
        image_result(payload)


def test_the_attribution_panel_is_absent_rather_than_empty_without_a_detector():
    """An empty panel reads as "nothing drove the verdict", a different claim from "no
    detector ran"."""
    assert "occlusion" not in image_result(_payload(), None)
    assert "occlusion" in image_result(
        _payload(), {"grid": 5, "image": "data:image/png;base64,AA",
                     "base_probability": 0.009, "peak": {"drop": 0.004}}
    )


def test_the_text_card_reports_the_threshold_the_verdict_used():
    """A verdict without its operating point cannot be checked by the reader."""
    html = text_result(
        arm_label="Arm B, matched mirrors", verdict="ai", probability=0.98, threshold=0.3125,
        fpr_budget=0.001, words=300, windows=3, maximum=0.99, model_version="x@abc1234",
        val_fnr=0.0043, val_ece=0.004,
    )
    assert "AI DETECTED" in html
    assert "0.312500" in html and "0.1%" in html
    assert "x@abc1234" in html
