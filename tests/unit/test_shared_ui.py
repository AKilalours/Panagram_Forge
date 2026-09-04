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


def _arm(name: str, label: str, verdict: str = "human", probability: float = 0.0001) -> dict:
    return {"arm": name, "label": label, "verdict": verdict, "ai_probability": probability,
            "max_window_probability": probability, "confidence": 1.0, "threshold": 0.992285,
            "fpr_budget": 0.001, "model_version": f"forge_min_{name}@4ec8204c",
            "abstained": False, "n_windows": 1, "windows": [probability],
            "val_fnr": 0.0043, "val_ece": 0.003713}


def _text_payload(**overrides) -> dict:
    base = {
        "available": True, "words": 211,
        "arms": [_arm("baseline", "A: random synthetic"), _arm("mirror", "B: matched mirrors")],
        "unavailable": {}, "caveat": "Trained on four generator families at 1.7B to 3.8B.",
        "abstention": "Off. The uncertain band is derived from validation scores.",
    }
    base.update(overrides)
    return base


def test_the_text_page_shows_both_arms_side_by_side():
    """The comparison IS the experiment. A single-arm page is a different product.

    An earlier draft of the Streamlit page showed one arm with a selector, to fit a 2.7 GB
    host. That silently dropped the finding the whole project exists to report.
    """
    html = text_result(_text_payload())
    assert "A: random synthetic" in html and "B: matched mirrors" in html
    assert "Both arms agree." in html
    assert "Mean over 1 window, 211 words." in html
    for label in ("Deployed threshold", "FPR budget", "Distance from threshold",
                  "Windows scored", "Its validation FNR", "Its validation ECE"):
        assert label in html, f"the arm card lost its {label!r} row"
    assert "What this score is not" in html and "Provenance" in html


def test_the_headline_is_the_mirror_arm_not_whichever_loaded_first():
    """A product has one verdict. The deployed arm is the headline; the other sits beside it."""
    arms = [_arm("baseline", "A", verdict="ai", probability=0.99),
            _arm("mirror", "B", verdict="human", probability=0.01)]
    html = text_result(_text_payload(arms=arms))
    headline = html[html.index("<h2>"):html.index("</h2>")]
    assert "NO AI DETECTED" in headline, "the banner followed arm A instead of the mirror arm"


def test_disagreement_is_stated_rather_than_averaged_away():
    """Two arms that disagree is a finding, not a presentation problem to smooth over."""
    arms = [_arm("baseline", "A: random synthetic", verdict="ai", probability=0.99),
            _arm("mirror", "B: matched mirrors", verdict="human", probability=0.01)]
    html = text_result(_text_payload(arms=arms))
    assert "THE ARMS DISAGREE" in html
    assert "Both arms agree" not in html


def test_one_arm_loaded_says_so_and_still_reports_the_other_as_unavailable():
    """On a small host the second arm may not fit. Saying so beats comparing an arm to itself."""
    html = text_result(_text_payload(
        arms=[_arm("mirror", "B: matched mirrors")],
        unavailable={"baseline": "not enough memory to hold a second arm"},
    ))
    assert "Only one arm is loaded." in html
    assert "baseline unavailable" in html
    assert "not enough memory" in html


def test_no_arm_loaded_produces_no_number_at_all():
    html = text_result(_text_payload(available=False, arms=[],
                                     unavailable={"mirror": "no checkpoint"}))
    assert "No verdict" in html
    assert "would be fabricated" in html
    assert "%" not in html.split("scorenote")[0].split("score ")[-1][:40]
