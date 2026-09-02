"""One normalisation, used by every stage that touches an image.

WHY THIS IS THE FIRST FILE IN THE IMAGE TRACK.

Real photographs arrive as JPEG: shot by a camera, resized by a hosting service, often
recompressed several times, carrying EXIF. Diffusion output arrives as a lossless PNG at
exactly 1024x1024, never compressed, never resized, with no camera pipeline behind it and
no EXIF at all.

A detector trained on those two populations reaches near-perfect accuracy WITHOUT EVER
LOOKING at whether an image was generated. It learns JPEG quantisation, resampling
signatures, aspect-ratio priors and the presence or absence of metadata. It then collapses
the first time someone screenshots an AI image or saves a real photograph as PNG.

That is the image-track equivalent of a text detector learning the topic, and it is worse,
because accuracy, AUROC and every other headline metric look excellent while it happens.

So: every image, human or generated, passes through THIS function before anything sees it.
One implementation, one policy object, no per-call-site variation. If ingestion and
generation normalise differently, the difference between them becomes the signal.

Nothing here is clever. Being boring and shared is the entire point.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

# Frozen with the spec. Changing any field changes what every model has ever seen, so it
# is a dataset-version change, not a tweak.
POLICY_VERSION = "image_norm_v1"


@dataclass(frozen=True)
class NormalizationPolicy:
    size: int = 512            # output is size x size, always square
    jpeg_quality: int = 90     # one quality for every image, human or generated
    subsampling: int = 2       # 4:2:0, the common web default
    version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        if self.size < 32:
            raise ValueError(f"size {self.size} is too small to be useful")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality {self.jpeg_quality} out of range")


POLICY_V1 = NormalizationPolicy()


class ImageTooSmallError(ValueError):
    """Upscaling invents detail, and invented detail is a generation artifact."""


def normalize_bytes(raw: bytes, policy: NormalizationPolicy = POLICY_V1) -> bytes:
    """Return JPEG bytes that carry no trace of how the input was encoded.

    Steps, in this order and for these reasons:

    1. Apply EXIF orientation, THEN discard EXIF. A phone photograph is often stored
       sideways with a rotation flag. Stripping the flag first would leave the crop taking
       the wrong region, which is a silent, systematic corruption of one class only.
    2. Convert to RGB. Alpha and palette channels appear in generated PNGs and almost never
       in photographs, so the channel count is itself a source signature.
    3. Centre crop to a square BEFORE resizing, so aspect ratio cannot survive as a
       feature. Photographs are 4:3 and 3:2; diffusion output is 1:1.
    4. Resize to the policy size with one resampling filter.
    5. Re-encode as JPEG at one fixed quality with fixed subsampling, no metadata.

    Refuses to upscale: an image smaller than the policy size is rejected rather than
    interpolated, because interpolation fabricates high-frequency content and that is
    precisely the kind of thing the detector is being asked to notice.
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)      # 1
        img = img.convert("RGB")                # 2

        short_side = min(img.size)
        if short_side < policy.size:
            raise ImageTooSmallError(
                f"image is {img.size[0]}x{img.size[1]}, short side {short_side} is below "
                f"the policy size {policy.size}. Upscaling would fabricate detail."
            )

        left = (img.size[0] - short_side) // 2
        top = (img.size[1] - short_side) // 2
        img = img.crop((left, top, left + short_side, top + short_side))   # 3
        img = img.resize((policy.size, policy.size), Image.LANCZOS)        # 4

        buf = io.BytesIO()
        img.save(                                                          # 5
            buf,
            format="JPEG",
            quality=policy.jpeg_quality,
            subsampling=policy.subsampling,
            optimize=False,
            progressive=False,
        )
    return buf.getvalue()


def describe(raw: bytes) -> dict:
    """Facts about an image, for the manifest. Never fed to the model in v1.

    Resolution and format are recorded because provenance needs them, and withheld from
    training because a model given the resolution would use it instead of the pixels.
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        return {
            "format": img.format,
            "mode": img.mode,
            "width": img.size[0],
            "height": img.size[1],
            "has_exif": bool(getattr(img, "_getexif", lambda: None)()),
        }
