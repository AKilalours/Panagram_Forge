"""The structured caption, which is the ONLY thing that crosses from a photograph to its mirror.

WHY A STRUCTURE AND NOT A SENTENCE. "A dog on a beach" produces a mirror that matches the
subject and nothing else: different composition, different light, different lens. The
detector then has an easy job for the wrong reason, because photographs and generations
still differ in everything except the noun. The text track learned the same lesson: a
mirror prompt that carries only the topic leaves the classifier free to learn the topic.

WHY NO PIXELS. The caption is a bottleneck by design. Nothing else may pass from the source
image to the generator: no latents, no img2img initialisation, no depth map, no edge map.
Under img2img the "mirror" inherits the original's structure and the detector ends up
learning a denoising signature rather than a generation signature, which is a different and
much less interesting experiment.

CAPTION_VERSION is part of the dataset identity. Changing a field, or the order fields are
rendered in, changes every prompt and therefore every generated image, so it is a dataset
version bump and not an edit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

CAPTION_VERSION = "image_mirror_v1"

# Fixed, ordered. The order is part of the prompt, so it is part of the version.
FIELDS = (
    "scene",
    "objects",
    "spatial_relations",
    "composition",
    "lighting",
    "colour_palette",
    "camera_perspective",
    "time_of_day",
)

MAX_FIELD_CHARS = 400

# A caption that names a person, a place or a brand pulls the generator toward reproducing
# a specific real image, which is contamination rather than matching, and it also drags
# identifiable information into a file FORGE publishes.
_IDENTIFYING = re.compile(
    r"\b(?:https?://|www\.)\S+"                    # URLs
    r"|\b[A-Z][a-z]+ [A-Z][a-z]+\b"                # Capitalised Full Names
    r"|\b(?:copyright|\(c\)|©)\b",
    re.IGNORECASE,
)


class CaptionRejected(ValueError):
    pass


@dataclass(frozen=True)
class Caption:
    scene: str
    objects: str
    spatial_relations: str
    composition: str
    lighting: str
    colour_palette: str
    camera_perspective: str
    time_of_day: str
    version: str = CAPTION_VERSION
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def validate(caption: Caption) -> None:
    """Reject a caption before it becomes 20,000 generated images.

    Every check here is cheap and every one of them is something that would otherwise be
    discovered after a GPU day.
    """
    for name in FIELDS:
        value = getattr(caption, name)
        if not isinstance(value, str) or not value.strip():
            raise CaptionRejected(
                f"field {name!r} is empty. A missing field silently drops one dimension of "
                "matching, so the mirror stops matching on it and nobody notices."
            )
        if len(value) > MAX_FIELD_CHARS:
            raise CaptionRejected(f"field {name!r} is {len(value)} chars, over {MAX_FIELD_CHARS}")
        if _IDENTIFYING.search(value):
            raise CaptionRejected(
                f"field {name!r} contains identifying text: {value[:80]!r}. Named people, "
                "places and URLs pull the generator toward reproducing a specific real "
                "image, which is contamination rather than matching."
            )


def to_prompt(caption: Caption) -> str:
    """Render the caption into a generation prompt, deterministically.

    One rendering function, so the prompt a mirror was generated from can always be
    reconstructed from the stored caption plus the version string.
    """
    validate(caption)
    parts = [f"{name.replace('_', ' ')}: {getattr(caption, name).strip()}" for name in FIELDS]
    return ". ".join(parts) + "."


class FakeCaptioner:
    """Deterministic stand-in so the mirror pipeline runs end to end without a VLM.

    It reads NOTHING about the image except its bytes, which is the point: it proves the
    plumbing without pretending to describe anything. Output is not a caption of the image
    and must never be used to build training data. The mirror engine records the captioner
    name on every record so fake-derived data cannot be mistaken for real.
    """

    name = "fake"
    version = "v1"

    _SCENES = ("an empty room", "a wide field", "a narrow street", "a still life on a table")
    _LIGHT = ("soft overcast light", "hard directional light", "warm low light", "flat even light")
    _ANGLES = ("eye level", "high angle", "low angle", "slightly above")

    def describe(self, image: bytes) -> Caption:
        import hashlib

        digest = hashlib.sha256(image).digest()
        pick = lambda seq, i: seq[digest[i] % len(seq)]  # noqa: E731
        return Caption(
            scene=pick(self._SCENES, 0),
            objects="two simple shapes and a plain background",
            spatial_relations="one shape left of the other, both near the centre",
            composition="centred subject with even margins",
            lighting=pick(self._LIGHT, 1),
            colour_palette="muted greys with one warm accent",
            camera_perspective=pick(self._ANGLES, 2),
            time_of_day="indeterminate",
            notes={"captioner": self.name, "captioner_version": self.version},
        )
