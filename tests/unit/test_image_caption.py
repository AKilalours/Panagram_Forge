"""The caption is the bottleneck, so its contract is the experiment's contract.

Everything that separates a matched mirror from an unmatched one passes through this
structure. A missing field silently drops one dimension of matching; an identifying string
pulls the generator toward reproducing a specific real photograph, which is contamination
rather than matching.
"""

from __future__ import annotations

import pytest

from forge.image.caption import (
    CAPTION_VERSION,
    FIELDS,
    Caption,
    CaptionRejected,
    FakeCaptioner,
    to_prompt,
    validate,
)


def _caption(**overrides) -> Caption:
    base = dict(
        scene="a quiet kitchen interior",
        objects="a kettle, two mugs, a folded cloth",
        spatial_relations="kettle left of the mugs, cloth in front",
        composition="off-centre subject, horizon low in frame",
        lighting="soft window light from the left",
        colour_palette="warm neutrals with a blue accent",
        camera_perspective="slightly above eye level, 35mm equivalent",
        time_of_day="mid morning",
    )
    base.update(overrides)
    return Caption(**base)


def test_a_complete_caption_validates() -> None:
    validate(_caption())


@pytest.mark.parametrize(
    "value",
    [
        "a quiet kitchen interior",
        "soft window light from the left",
        "warm neutrals with a blue accent",
        "slightly above eye level, 35mm equivalent",
        "two people seated at a table",
        "mid morning",
    ],
)
def test_ordinary_prose_is_accepted(value: str) -> None:
    """The counterpart every rejection rule needs.

    The proper-noun check works by capitalisation and was once compiled with IGNORECASE,
    which made it match any two lowercase words, so it rejected every caption ever written.
    A check that fires on everything is as useless as one that fires on nothing, and only a
    test of the ACCEPTING side catches that.
    """
    validate(_caption(scene=value, objects=value, lighting=value))


@pytest.mark.parametrize("missing", FIELDS)
def test_every_field_is_required(missing: str) -> None:
    """A blank field is worse than a missing one: matching quietly stops on that axis."""
    with pytest.raises(CaptionRejected, match=missing):
        validate(_caption(**{missing: "   "}))


@pytest.mark.parametrize(
    "value",
    [
        "Jane Doe standing in a kitchen",
        "shot in Golden Gate park",
        "a Coca Cola bottle on the table",
    ],
)
def test_identifying_names_are_rejected(value: str) -> None:
    with pytest.raises(CaptionRejected, match="identifying"):
        validate(_caption(scene=value))


def test_urls_are_rejected() -> None:
    with pytest.raises(CaptionRejected, match="identifying"):
        validate(_caption(objects="see https://example.com/photo.jpg"))


def test_copyright_markers_are_rejected() -> None:
    with pytest.raises(CaptionRejected, match="identifying"):
        validate(_caption(objects="a mug, copyright someone"))


def test_overlong_fields_are_rejected() -> None:
    with pytest.raises(CaptionRejected):
        validate(_caption(scene="x" * 5000))


def test_prompt_contains_every_field() -> None:
    prompt = to_prompt(_caption())
    for name in FIELDS:
        assert getattr(_caption(), name) in prompt


def test_prompt_is_deterministic() -> None:
    """The prompt an image came from must be reconstructible from the stored caption."""
    assert to_prompt(_caption()) == to_prompt(_caption())


def test_prompt_field_order_is_fixed() -> None:
    """Order is part of the prompt, so it is part of the dataset version."""
    prompt = to_prompt(_caption())
    positions = [prompt.index(getattr(_caption(), name)) for name in FIELDS]
    assert positions == sorted(positions)


def test_an_invalid_caption_cannot_become_a_prompt() -> None:
    """Validation must sit on the path to generation, not beside it."""
    with pytest.raises(CaptionRejected):
        to_prompt(_caption(lighting=""))


def test_caption_is_versioned() -> None:
    assert _caption().version == CAPTION_VERSION


def test_fake_captioner_is_deterministic_and_valid() -> None:
    captioner = FakeCaptioner()
    first = captioner.describe(b"some image bytes")
    second = captioner.describe(b"some image bytes")
    assert first == second
    validate(first)


def test_fake_captioner_varies_with_the_image() -> None:
    captioner = FakeCaptioner()
    captions = {captioner.describe(bytes([i]) * 64).scene for i in range(40)}
    assert len(captions) > 1


def test_fake_captioner_marks_itself() -> None:
    """Fake-derived data must never be mistakeable for real."""
    assert FakeCaptioner().describe(b"x").notes["captioner"] == "fake"
