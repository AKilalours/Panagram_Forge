"""The control arm must differ from the mirror arm in exactly one respect.

Same generators, same pinned revisions, same generation spec, same normalisation, same
count. The only permitted difference is where the prompt came from: a mirror describes a
specific photograph, a control is drawn from a fixed inventory that no photograph informed.

The text track came within one config field of failing this: both arms pointed at the same
directory, so they would have trained on identical data, produced identical numbers, and
supported the conclusion "matching makes no difference". Nothing would have errored.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")

from forge.common.splits import assign_split  # noqa: E402
from forge.image.caption import validate as validate_caption  # noqa: E402
from forge.image.generators import FakeImageGenerator, GenerationSpec  # noqa: E402
from forge.image.normalize import POLICY_V1, describe  # noqa: E402
from forge.image.unmatched import (  # noqa: E402
    SCENES,
    UNMATCHED_VERSION,
    assert_arms_comparable,
    generate_unmatched,
    spec_for,
)

SPEC = GenerationSpec(steps=2, guidance=2.0, resolution=512, seed=3)


def _sim(value: float):
    return lambda *_: value


# --- the inventory ---------------------------------------------------------------------


def test_every_inventory_caption_is_valid() -> None:
    """An invalid caption here would fail thousands of images deep into a paid run."""
    for i in range(200):
        validate_caption(spec_for(i))


def test_captions_are_deterministic() -> None:
    assert spec_for(37) == spec_for(37)


def test_the_inventory_produces_many_distinct_captions() -> None:
    """Fields moving in lockstep would give twelve combinations, not thousands."""
    combos = {tuple(spec_for(i).as_dict()[f] for f in ("scene", "objects", "colour_palette",
                                                       "composition", "time_of_day"))
              for i in range(400)}
    assert len(combos) > len(SCENES) * 4


def test_captions_describe_ordinary_scenes() -> None:
    """A deliberately weak control proves nothing, so the inventory must be plausible."""
    assert len(SCENES) >= 8
    assert all(len(scene[0].split()) >= 4 for scene in SCENES)


# --- the arm --------------------------------------------------------------------------


def test_control_images_are_produced_and_normalised() -> None:
    records, images, stats = generate_unmatched(
        6, FakeImageGenerator(), caption_similarity=_sim(0.5), spec=SPEC
    )
    assert stats.accepted == 6
    for _, data in images:
        got = describe(data)
        assert got["format"] == "JPEG"
        assert (got["width"], got["height"]) == (POLICY_V1.size, POLICY_V1.size)


def test_control_records_are_marked_unmatched() -> None:
    """The field the training loader checks to prove the arms are not the same data."""
    records, _, _ = generate_unmatched(
        3, FakeImageGenerator(), caption_similarity=_sim(0.5), spec=SPEC
    )
    for record in records:
        assert record.prompt_version == UNMATCHED_VERSION
        assert record.topic_match is False
        assert record.composition_match is False
        assert record.lighting_match is False
        assert record.notes["arm"] == "unmatched"


def test_control_images_have_no_source_image() -> None:
    """By construction. A control that referenced a photograph would not be a control."""
    records, _, _ = generate_unmatched(
        3, FakeImageGenerator(), caption_similarity=_sim(0.5), spec=SPEC
    )
    assert all(record.source_image_id == "" for record in records)


def test_control_images_get_their_own_group_and_split() -> None:
    records, _, _ = generate_unmatched(
        40, FakeImageGenerator(), caption_similarity=_sim(0.5), spec=SPEC
    )
    for record in records:
        assert record.source_group_id.startswith("grp_")
        assert record.split == assign_split(record.source_group_id).value
    assert len({r.split for r in records}) > 1, "every control image landed in one split"


def test_prompt_adherence_failures_are_counted() -> None:
    _, _, stats = generate_unmatched(
        5, FakeImageGenerator(), caption_similarity=_sim(0.0), spec=SPEC
    )
    assert stats.accepted == 0
    assert stats.rejected["ignored_prompt"] == 5


def test_generation_failure_is_recorded_not_swallowed() -> None:
    class Broken(FakeImageGenerator):
        def generate(self, prompts, spec):
            raise RuntimeError("cuda gone")

    _, _, stats = generate_unmatched(
        4, Broken(), caption_similarity=_sim(0.5), spec=SPEC, batch_size=2
    )
    assert stats.rejected["generation_failed"] == 4


def test_generation_is_batched() -> None:
    sizes: list[int] = []

    class Counting(FakeImageGenerator):
        def generate(self, prompts, spec):
            sizes.append(len(prompts))
            return super().generate(prompts, spec)

    generate_unmatched(
        20, Counting(), caption_similarity=_sim(0.5), spec=SPEC, batch_size=8
    )
    assert max(sizes) > 1 and max(sizes) <= 8


def test_records_carry_the_same_provenance_fields_as_mirrors() -> None:
    """The two arms must be comparable row by row, or the ablation cannot be tabulated."""
    from forge.image.mirror import ImageMirror

    records, _, _ = generate_unmatched(
        1, FakeImageGenerator(), caption_similarity=_sim(0.5), spec=SPEC
    )
    assert isinstance(records[0], ImageMirror)
    assert records[0].generation == SPEC.as_dict()
    assert records[0].generator_revision


# --- the guard against the failure the text track nearly shipped ------------------------


def test_identical_specs_are_accepted() -> None:
    assert_arms_comparable(SPEC, GenerationSpec(steps=2, guidance=2.0, resolution=512, seed=3))


@pytest.mark.parametrize(
    "difference",
    [{"steps": 40}, {"guidance": 9.0}, {"resolution": 768}, {"scheduler": "other"}],
)
def test_a_difference_between_arms_is_refused(difference: dict) -> None:
    """Any of these would make a win attributable to the setting rather than to matching."""
    other = GenerationSpec(**{**SPEC.as_dict(), **difference})
    with pytest.raises(RuntimeError, match="attributable"):
        assert_arms_comparable(SPEC, other)
