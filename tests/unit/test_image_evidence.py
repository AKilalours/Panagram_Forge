"""How the evidence panel is allowed to describe a stream.

The bug this guards: a stream's `strength` is how much that stream has to say, not a
probability of anything. Shown as a percentage next to a bar it is read as confidence, and
"Camera metadata 50%" is read as "50% likely to be a photograph". The panel now shows a
word and keeps the number in the payload.
"""

from __future__ import annotations


def test_a_stream_reports_a_word_rather_than_a_percentage():
    """`strength` is how much a stream has to say, not a probability.

    Rendered as "50%" beside a camera-metadata bar it reads as "50% likely to be a
    photograph", which is not what it means and is not recoverable from a caption. The
    panel shows `state`; the number stays in the payload.
    """
    from forge.image.evidence import NEUTRAL, Stream

    def s(strength, available=True):
        return Stream("k", "L", strength, NEUTRAL, "", available=available)

    assert s(0, available=False).state == "not available"
    assert s(0).state == "not detected"
    assert s(15).state == "weak"
    assert s(50).state == "partial"
    assert s(85).state == "detected"


def test_an_unavailable_stream_is_never_reported_as_detected():
    """The untrained visual model must not read as a signal that fired."""
    from forge.image.evidence import NEUTRAL, Stream

    for strength in (0, 40, 100):
        assert Stream("k", "L", strength, NEUTRAL, "", available=False).state == "not available"


def test_the_state_travels_in_the_payload():
    from forge.image.evidence import NEUTRAL, Stream

    assert Stream("k", "L", 85, NEUTRAL, "").as_dict()["state"] == "detected"
