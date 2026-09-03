"""The image detector: a real model, or nothing.

This is the stream that decides AI or not AI for an image. Everything else in the image
report describes the file and cannot answer the question.

**FORGE-Image is not trained yet, so this loads a published baseline detector.** That is a
deliberate, stated position, not a disguise:

  A baseline is required anyway. "Failure-driven mirroring improves image detection" is
  unfalsifiable without something to improve on, and every detection result in the text
  track is reported against a control arm for the same reason.

  A baseline is a model. Its probability comes from pixels through learned weights, which
  is categorically different from a number assembled out of EXIF and quantisation tables.
  The second would fire hardest on screenshots and re-saved photographs, and this project
  exists to keep innocent images out of that trap.

  It is labelled. The report carries the model id, says the probability is UNCALIBRATED,
  and says the detector is a third-party baseline rather than FORGE-Image. Presenting it as
  FORGE's own trained detector would be the dishonest version; refusing to run a baseline at
  all is not more honest, only less useful.

**LABEL POLARITY IS RESOLVED, NEVER GUESSED.** A published classifier may order its classes
either way, and the same mistake in `parse_mage` would have turned an AUROC of 0.05 into a
reported 0.95. The class names are matched against an explicit table and a model whose
labels are not recognised is REFUSED, with its labels in the error, rather than defaulting
to index 1.

**The probability is not calibrated for this project's operating point.** Calibration needs
a validation split from a corpus that does not exist yet, so the number shown is the model's
own output and the report says so. The decision band is the documented default below, not a
fitted threshold, and it is stated wherever the verdict appears.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache

# Tried in order. Overridable with FORGE_IMAGE_DETECTOR, which takes priority and is tried
# alone: a caller who names a model wants that model, not a silent fallback to another one.
CANDIDATES: tuple[str, ...] = (
    "Organika/sdxl-detector",
    "umm-maybe/AI-image-detector",
    "haywoodsloan/ai-image-detector-deploy",
)

# Class-name to meaning. Substring match on a lowercased label, longest first so that
# "not ai" is read before "ai". Anything unmatched is a refusal, not a default.
AI_WORDS = ("artificial", "ai-generated", "ai generated", "generated", "fake", "synthetic",
            "midjourney", "stable diffusion", "sdxl", "dalle", "ai")
HUMAN_WORDS = ("not ai", "non-ai", "human", "real", "authentic", "photo", "camera", "natural")

# Documented default, NOT fitted. Stated everywhere the verdict is shown.
ABSTAIN_LOW = 0.20
ABSTAIN_HIGH = 0.80


class DetectorUnavailable(RuntimeError):
    """Raised with a reason a reader can act on. Never swallowed into a score."""


class LabelPolarityUnknown(DetectorUnavailable):
    """The model's classes could not be mapped to AI and not-AI."""


@dataclass(frozen=True)
class Detection:
    ai_probability: float
    model_id: str
    calibrated: bool = False
    labels: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        if self.ai_probability >= ABSTAIN_HIGH:
            return "ai"
        if self.ai_probability < ABSTAIN_LOW:
            return "human"
        return "uncertain"


def resolve_ai_index(id2label: dict) -> int:
    """Which output index means AI-generated? Matched by name, or refused.

    Returns the index. Raises LabelPolarityUnknown when the labels do not resolve to exactly
    one AI class and at least one non-AI class, with the labels in the message.

    The failure this prevents: assuming index 1 is the positive class. A model ordering its
    classes the other way would invert every verdict on the page while looking completely
    normal, which is the same bug that reversed the MAGE benchmark.
    """
    def meaning(label: str) -> str | None:
        text = str(label).strip().lower().replace("_", " ")
        for word in sorted(HUMAN_WORDS, key=len, reverse=True):
            if word in text:
                return "human"
        for word in sorted(AI_WORDS, key=len, reverse=True):
            if word in text:
                return "ai"
        return None

    resolved = {int(i): meaning(label) for i, label in id2label.items()}
    ai = [i for i, m in resolved.items() if m == "ai"]
    human = [i for i, m in resolved.items() if m == "human"]
    if len(ai) != 1 or not human:
        raise LabelPolarityUnknown(
            f"cannot tell which class means AI from labels {dict(id2label)}. Refusing rather "
            "than assuming an index: a wrong guess inverts every verdict silently."
        )
    return ai[0]


@dataclass
class ImageDetector:
    model_id: str
    ai_index: int
    labels: tuple[str, ...]
    model: object
    processor: object

    def probability(self, data: bytes) -> float:
        import torch
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            rgb = img.convert("RGB")
            inputs = self.processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return float(torch.softmax(logits.float(), dim=-1)[0, self.ai_index])

    def detect(self, data: bytes) -> Detection:
        return Detection(
            ai_probability=self.probability(data),
            model_id=self.model_id,
            calibrated=False,
            labels=self.labels,
        )


def _load_one(model_id: str) -> ImageDetector:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    # float32 on CPU. A half-precision checkpoint fails at the first matmul with
    # "mat1 and mat2 must have the same dtype", which is a total failure, not a slow one.
    model.float()
    model.eval()

    id2label = getattr(model.config, "id2label", None) or {}
    if len(id2label) < 2:
        raise LabelPolarityUnknown(f"{model_id} exposes no usable labels: {id2label}")
    return ImageDetector(
        model_id=model_id,
        ai_index=resolve_ai_index(id2label),
        labels=tuple(str(v) for _, v in sorted(id2label.items(), key=lambda kv: int(kv[0]))),
        model=model,
        processor=processor,
    )


@lru_cache(maxsize=1)
def load_detector() -> ImageDetector:
    """Load the image detector, or raise with every attempt's reason.

    Cached: a per-request load of a few hundred megabytes is not serving.
    """
    named = os.getenv("FORGE_IMAGE_DETECTOR", "").strip()
    if named:
        # Named explicitly: try that and only that. A silent fallback to a different model
        # would mean the report names one detector and a different one produced the score.
        return _load_one(named)

    failures = []
    for model_id in CANDIDATES:
        try:
            return _load_one(model_id)
        except Exception as error:  # noqa: BLE001 - collected and reported together
            failures.append(f"{model_id}: {type(error).__name__}: {error}")
    raise DetectorUnavailable(
        "no image detector could be loaded. Set FORGE_IMAGE_DETECTOR to a model id, or "
        "train FORGE-Image. Attempts: " + " | ".join(failures)
    )


def detector_state() -> dict:
    """Whether a detector can serve, and if not, why. Never raises."""
    try:
        detector = load_detector()
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "model_id": detector.model_id,
        "labels": list(detector.labels),
        "ai_index": detector.ai_index,
        "calibrated": False,
    }
