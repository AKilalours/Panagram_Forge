"""A mirror must be matched, and the two ways it can fail to be are opposite.

TOO DISSIMILAR: the generator ignored the prompt, so the pair is a random photograph beside
a random generation. That is the control arm. Training on it while calling it a mirror
turns arm B into a second copy of arm A, and the experiment then measures nothing while
producing two plausible rows.

TOO SIMILAR: the generator reproduced something close to a memorised image, so the AI
example contains the human example's content and the detector is asked to separate an image
from itself.

A single threshold catches one of these. The validator needs a band, which is the image
analogue of the text validator's length-ratio window.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")
from PIL import Image  # noqa: E402

from forge.common.splits import assign_split  # noqa: E402
from forge.image.caption import Caption, FakeCaptioner  # noqa: E402
from forge.image.generators import FakeImageGenerator, GenerationSpec  # noqa: E402
from forge.image.mirror import (  # noqa: E402
    ValidityPolicy,
    assert_split_inheritance,
    check_validity,
    generate_mirrors,
)
from forge.image.normalize import POLICY_V1, describe  # noqa: E402

SPEC = GenerationSpec(steps=2, guidance=2.0, resolution=512, seed=7)


def _photo(seed: int, w: int = 700, h: int = 700) -> bytes:
    period = 3 + (seed % 5)
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data += bytes(
                (((x // period) * 47 + seed) % 256, ((y // 4) * 29 + seed) % 256, (x + y) % 256)
            )
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), bytes(data)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _sources(n: int) -> list[tuple[str, bytes]]:
    return [(f"oi_{i:05d}", _photo(i + 1)) for i in range(n)]


def _sim(value: float):
    return lambda *_: value


def _caption() -> Caption:
    return Caption(
        scene="a plain interior",
        objects="two shapes",
        spatial_relations="side by side",
        composition="centred",
        lighting="even light",
        colour_palette="muted greys",
        camera_perspective="eye level",
        time_of_day="indeterminate",
    )


# --- the validity band -----------------------------------------------------------------


def test_a_matched_candidate_is_accepted() -> None:
    assert check_validity(b"x", b"y", "p", ValidityPolicy(), _sim(0.7), _sim(0.5)) is None


def test_a_candidate_that_ignored_the_prompt_is_rejected() -> None:
    reason = check_validity(b"x", b"y", "p", ValidityPolicy(), _sim(0.7), _sim(0.01))
    assert reason == "ignored_prompt"


def test_an_unrelated_candidate_is_rejected() -> None:
    """Otherwise arm B silently becomes a second copy of arm A."""
    reason = check_validity(b"x", b"y", "p", ValidityPolicy(), _sim(0.10), _sim(0.5))
    assert reason == "not_a_mirror"


def test_a_reproduction_of_the_source_is_rejected_separately() -> None:
    """Counted on its own: a rising rate means the generator is reciting training data."""
    reason = check_validity(b"x", b"y", "p", ValidityPolicy(), _sim(0.99), _sim(0.5))
    assert reason == "reproduced_source"


def test_an_inverted_band_is_refused_at_construction() -> None:
    """A band with min above max would accept nothing, or everything, silently."""
    with pytest.raises(ValueError):
        ValidityPolicy(source_similarity_min=0.9, source_similarity_max=0.4)


# --- the pipeline ----------------------------------------------------------------------


def test_mirrors_are_produced_and_normalised() -> None:
    records, images, stats = generate_mirrors(
        _sources(4), FakeCaptioner(), FakeImageGenerator(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    assert stats.accepted == 4
    assert len(records) == len(images) == 4
    for _, data in images:
        got = describe(data)
        assert got["format"] == "JPEG"
        assert (got["width"], got["height"]) == (POLICY_V1.size, POLICY_V1.size)


def test_generated_images_go_through_the_same_normalisation_as_photographs() -> None:
    """The confound the image track exists to avoid.

    Generators emit lossless PNG at their native resolution; photographs arrive as JPEG.
    If the two populations were stored differently, the detector would learn the encoder
    and every headline metric would look excellent while it did.
    """
    _, images, _ = generate_mirrors(
        _sources(1), FakeCaptioner(), FakeImageGenerator(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    raw_generated = FakeImageGenerator().generate(["p"], SPEC)[0]
    assert describe(raw_generated)["format"] == "PNG"      # what the generator emits
    assert describe(images[0][1])["format"] == "JPEG"      # what FORGE stores


def test_a_mirror_inherits_its_source_split() -> None:
    sources = _sources(30)
    records, _, _ = generate_mirrors(
        sources, FakeCaptioner(), FakeImageGenerator(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    assert records
    for record in records:
        assert record.split == assign_split(record.source_image_id).value
    assert_split_inheritance(sources, records)


def test_split_inheritance_is_checked_not_trusted() -> None:
    records, _, _ = generate_mirrors(
        _sources(4), FakeCaptioner(), FakeImageGenerator(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    broken = [type(records[0])(**{**records[0].as_dict(), "split": "nowhere"})]
    with pytest.raises(RuntimeError, match="inflated metrics"):
        assert_split_inheritance(_sources(4), broken)


def test_rejections_are_counted_by_reason() -> None:
    _, _, stats = generate_mirrors(
        _sources(5), FakeCaptioner(), FakeImageGenerator(),
        image_similarity=_sim(0.99), caption_similarity=_sim(0.5), spec=SPEC,
    )
    assert stats.accepted == 0
    assert stats.rejected["reproduced_source"] == 5
    assert stats.as_dict()["acceptance_rate"] == 0.0


def test_a_failing_generator_is_recorded_not_swallowed() -> None:
    class Broken:
        family, model_id, revision = "broken", "x", "a" * 40

        def generate(self, prompts, spec):
            raise RuntimeError("cuda gone")

        def close(self):
            return None

    _, _, stats = generate_mirrors(
        _sources(3), FakeCaptioner(), Broken(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    assert stats.rejected["generation_failed"] == 3


def test_an_invalid_caption_stops_before_any_gpu_work() -> None:
    class BadCaptioner:
        name = "bad"

        def describe(self, image: bytes) -> Caption:
            return Caption(**{**_caption().as_dict(), "lighting": ""})

    called = {"n": 0}

    class Counting(FakeImageGenerator):
        def generate(self, prompts, spec):
            called["n"] += 1
            return super().generate(prompts, spec)

    _, _, stats = generate_mirrors(
        _sources(3), BadCaptioner(), Counting(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    assert stats.rejected["caption_rejected"] == 3
    assert called["n"] == 0, "generation ran despite every caption being invalid"


def test_records_carry_full_generation_provenance() -> None:
    records, _, _ = generate_mirrors(
        _sources(1), FakeCaptioner(), FakeImageGenerator(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC,
    )
    record = records[0]
    assert record.generator_revision
    assert record.generation == SPEC.as_dict()
    assert set(record.generation) >= {"steps", "guidance", "resolution", "scheduler", "seed"}
    assert record.caption["scene"]
    assert record.notes["captioner"] == "fake"


def test_generation_is_batched() -> None:
    """One prompt per call wastes most of the GPU; the text track measured that cost."""
    sizes: list[int] = []

    class Counting(FakeImageGenerator):
        def generate(self, prompts, spec):
            sizes.append(len(prompts))
            return super().generate(prompts, spec)

    generate_mirrors(
        _sources(20), FakeCaptioner(), Counting(),
        image_similarity=_sim(0.7), caption_similarity=_sim(0.5), spec=SPEC, batch_size=8,
    )
    assert max(sizes) > 1 and max(sizes) <= 8
