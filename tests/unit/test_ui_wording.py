"""What the interface is allowed to say.

Two rules, both about claims the product cannot support:

  A detector cannot establish that a person made something. It reports whether the input
  resembles the AI distribution it was trained on. "HUMAN" is a claim about the world;
  "NO AI DETECTED" is a claim about the model, and only the second one is true.

  Version identifiers belong in the payload, not on the face of the product. "FORGE-Image
  v0.4" beside a verdict reads as a released product to anyone who does not know the
  project, and the string changes every run.
"""

from __future__ import annotations

import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[2] / "api" / "forge_app.py"


def page() -> str:
    return PAGE.read_text()


def test_the_verdict_word_is_no_ai_detected_not_human():
    body = page()
    assert "NO AI DETECTED" in body
    assert "AI DETECTED" in body
    assert "UNCERTAIN" in body
    assert re.search(r"WORD\s*=\s*\{[^}]*human:\s*'NO AI DETECTED'", body), (
        "the verdict map must translate the model's 'human' label before display"
    )


def test_no_version_identifier_is_rendered_beside_a_verdict_or_a_score():
    """Provenance lives in the technical footer, not in the headline panels."""
    body = page()
    banner = body[body.index("function banner("):body.index("function renderImage(")]
    for forbidden in ("model_version", "report_version", "v0.", "version"):
        assert forbidden not in banner, f"the banner renders {forbidden!r}"


def test_provenance_is_still_reachable_rather_than_deleted():
    """Demoted, not removed. A detector with no identifiable build cannot be audited."""
    body = page()
    assert "model_version" in body, "provenance was dropped entirely"
    assert "for reproducibility rather than" in body


def test_the_header_states_system_state_without_a_build_number():
    body = page()
    header = body[body.index("let s={}, img={};"):body.index("const drop=$('#drop')")]
    assert "image detector " in header
    # Only the rendered string matters; the endpoint path may legitimately mention arms.
    shown = header[header.index("txt.textContent"):header.index("dot.className")]
    assert "v0." not in shown and "version" not in shown
    assert "arms" not in shown, "the arm count is a developer readout, not product state"


def test_the_primary_evidence_stream_is_the_detector_and_is_named_not_guessed():
    """The hierarchy depends on one key. A silent fallback would invert it.

    An earlier version looked up 'visual' and fell back to streams[0]. That happened to be
    the right stream, so a wrong key rendered a correct panel: the class of bug that only
    shows up when the order changes. If the detector stream is absent the panel says so
    rather than promoting camera metadata to PRIMARY evidence.
    """
    body = page()
    assert "s.key==='visual_model'" in body
    assert "ev.streams[0]" not in body, "a positional fallback for the primary stream is back"


def test_the_key_the_page_looks_up_is_the_key_the_evidence_engine_emits():
    """Pins the contract across the two files, so a rename fails here, not in the browser."""
    import io

    import numpy as np
    import pytest

    pytest.importorskip("PIL")
    from PIL import Image

    from forge.image.report import build_report

    buffer = io.BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (160, 120, 3), dtype="uint8")).save(
        buffer, format="JPEG", quality=90
    )
    streams = build_report(buffer.getvalue(), with_stability=False).as_dict()["evidence"]["streams"]
    assert any(s["key"] == "visual_model" for s in streams), (
        f"the page looks up 'visual_model'; the engine emits {[s['key'] for s in streams]}"
    )


def test_development_language_is_off_the_production_surface():
    """The interface states what the system does, not the argument behind the decision."""
    from forge.image.report import Assessment

    reason = Assessment().reason
    for phrase in ("architecturally complete", "has not been trained", "fabricated"):
        assert phrase not in reason.lower(), f"{phrase!r} is still in the headline"
    assert "Visual detector unavailable" in reason
    assert Assessment().detail, "the longer explanation was deleted rather than moved"


def test_the_working_detector_is_the_tab_that_opens_first():
    """Text is trained and calibrated; the image detector does not exist.

    Opening on Image made the first thing anyone saw "No verdict", which reads as "the
    product does not work" while a working detector sat one click away.
    """
    body = page()
    start = body.index('<div class="tabs">')
    tabs = body[start:body.index("</div>\n  <div class=\"status\"", start)]
    assert tabs.index('data-tab="text"') < tabs.index('data-tab="image"')
    assert '<div class="tab on" data-tab="text">' in body
    assert '<section id="pane-image" class="hide">' in body


def test_the_diagnostics_are_collapsed_and_the_verdict_is_not():
    """Product question first, research second. Collapsed, never deleted."""
    body = page()
    for section in ("Forensic diagnostics", "Performance details",
                    "Limitations and interpretation"):
        assert f"<summary>{section}" in body, f"{section} is not a collapsed section"
    assert "details class=\"card wide fold\"" in body
    # The banner and the evidence panel must not be inside a fold.
    render = body[body.index("function renderImage(d){"):body.index("$('#go').onclick")]
    head = render[:render.index("<details")]
    assert "banner({" in head and "Evidence" in head
