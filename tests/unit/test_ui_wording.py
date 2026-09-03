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
    header = body[body.index("let s={}; try{"):body.index("const drop=$('#drop')")]
    assert "image detector not trained" in header
    # Only the rendered string matters; the endpoint path may legitimately mention arms.
    shown = header[header.index("txt.textContent"):header.index("dot.className")]
    assert "v0." not in shown and "version" not in shown
    assert "arms" not in shown, "the arm count is a developer readout, not product state"
