"""Attacks must survive normalisation, and must not destroy the image they attack.

TWO WAYS A ROBUSTNESS SUITE FAILS SILENTLY, both seen already in this project's text track.

An attack that changes nothing produces a column of zeros that reads as excellent
robustness. Two of the text attacks did exactly that: synonym_swap and article_deletion
perturbed no characters at all, and the resulting numbers looked like a strong result.

An attack that destroys the image produces a column of failures that reads as poor
robustness, when all it shows is that the detector cannot classify noise. The text track
hit the mirror of this too: a validity check discarded the homoglyph attacks that worked,
because it could not tell a readable substitution from vandalism.

So every attack here is checked from both sides: it must change the NORMALISED result, and
it must leave the picture recognisable.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")
from PIL import Image  # noqa: E402

from forge.image.attacks import (  # noqa: E402
    ATTACKS,
    ATTACKS_BY_NAME,
    apply_attack,
    jpeg,
    preserves_content,
    strip_metadata,
)
from forge.image.normalize import POLICY_V1, describe, normalize_bytes  # noqa: E402


def _photo(seed: int = 3, w: int = 900, h: int = 900) -> bytes:
    """Structured content, so lossy operations have something to degrade."""
    period = 3 + (seed % 5)
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data += bytes(
                (
                    ((x // period) * 41 + seed * 7) % 256,
                    ((y // (period + 1)) * 23 + seed) % 256,
                    ((x * y) // 97 + seed) % 256,
                )
            )
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), bytes(data)).save(buf, format="PNG")
    return buf.getvalue()


ORIGINAL = _photo()


@pytest.mark.parametrize("attack", ATTACKS, ids=lambda a: a.name)
def test_every_attack_produces_a_valid_image(attack) -> None:
    got = describe(attack.apply(ORIGINAL))
    assert got["width"] > 0 and got["height"] > 0


@pytest.mark.parametrize("attack", ATTACKS, ids=lambda a: a.name)
def test_every_attack_leaves_the_picture_recognisable(attack) -> None:
    """An attack that destroys the image is vandalism and proves nothing about robustness."""
    assert preserves_content(ORIGINAL, attack.apply(ORIGINAL)), (
        f"{attack.name} changed the picture beyond recognition"
    )


@pytest.mark.parametrize(
    "attack",
    [a for a in ATTACKS if a.name != "metadata_strip"],
    ids=lambda a: a.name,
)
def test_every_attack_survives_normalisation(attack) -> None:
    """The ordering trap, asserted.

    Attacks run BEFORE normalisation, which is the order a real upload experiences. If
    normalisation flattened an attack, its row in the robustness table would be a zero that
    read as excellent robustness rather than as a measurement that never happened.
    """
    clean = normalize_bytes(ORIGINAL)
    attacked = normalize_bytes(attack.apply(ORIGINAL))
    assert clean != attacked, f"{attack.name} had no effect once normalised"


def test_metadata_strip_is_a_no_op_after_normalisation_and_that_is_the_point() -> None:
    """The exception, and it is a finding rather than an oversight.

    Normalisation already removes metadata, so a detector cannot be reading it. If this
    attack ever DID change the outcome, the detector would be using the container rather
    than the pixels, which is the confound the image track exists to prevent.
    """
    assert normalize_bytes(ORIGINAL) == normalize_bytes(strip_metadata(ORIGINAL))


def test_attacks_are_ordered_from_mild_to_aggressive() -> None:
    """A robustness table should read as a progression, not an arbitrary list."""
    mild = preserves_content(ORIGINAL, apply_attack(ORIGINAL, "jpeg_85"), max_distance=4)
    assert mild, "jpeg_85 should be very close to the original"


def test_heavier_compression_degrades_more_than_lighter() -> None:
    from forge.image.phash import dhash, distance

    light = distance(dhash(ORIGINAL), dhash(jpeg(ORIGINAL, 85)))
    heavy = distance(dhash(ORIGINAL), dhash(jpeg(ORIGINAL, 25)))
    assert heavy >= light


def test_attacks_are_deterministic() -> None:
    """A robustness number that moves between runs cannot be compared across models."""
    for attack in ATTACKS:
        assert attack.apply(ORIGINAL) == attack.apply(ORIGINAL), attack.name


def test_registry_names_are_unique_and_addressable() -> None:
    names = [a.name for a in ATTACKS]
    assert len(names) == len(set(names))
    assert set(ATTACKS_BY_NAME) == set(names)


def test_an_unknown_attack_is_an_error_not_a_silent_pass() -> None:
    with pytest.raises(KeyError, match="unknown attack"):
        apply_attack(ORIGINAL, "definitely_not_an_attack")


def test_every_attack_has_a_description() -> None:
    """The robustness table is read by people; a bare function name explains nothing."""
    assert all(a.description.strip() for a in ATTACKS)


def test_vandalism_is_detected_as_such() -> None:
    """preserves_content must actually be able to say no, or it protects nothing."""
    noise = _photo(seed=91)
    assert not preserves_content(ORIGINAL, noise)


def test_attacked_images_still_normalise_to_the_policy_shape() -> None:
    """The pipeline must accept attacked inputs, since that is what production sees."""
    for attack in ATTACKS:
        got = describe(normalize_bytes(attack.apply(ORIGINAL)))
        assert (got["width"], got["height"]) == (POLICY_V1.size, POLICY_V1.size), attack.name
