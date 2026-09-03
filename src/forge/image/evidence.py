"""The evidence engine: turning forensic findings into the panels a person reads.

THE ARCHITECTURAL DECISION THIS FILE ENCODES.

The obvious design collapses every signal into one authenticity probability. It is the
wrong design and it fails in a specific, predictable way.

Consider a holiday photograph sent over WhatsApp. Metadata stripped, re-encoded with
library quantisation tables, resized. Every "authenticity" signal reads negative. A combined
score calls it suspicious. Now consider an image from a generator that writes no metadata:
identical signal profile. The score cannot separate them, because the signals describe HOW
THE FILE WAS WRITTEN, not what is in it.

So the engine does three things instead of one:

  1. Reports each evidence stream at its own strength, with the direction it points.
  2. Detects CONFLICT between streams, which is the genuinely informative case. A file that
     declares itself AI-generated while carrying a full camera EXIF block is interesting.
     Two streams agreeing tells you far less than two disagreeing.
  3. Refuses to emit an overall verdict from metadata alone. That is the detector's job.

WHAT EACH STREAM MEASURES, and note that only one of them points at "AI".

  visual_model       the trained detector. Absent until FORGE-Image is trained.
  camera_metadata    how complete and internally consistent the capture metadata is.
                     HIGH means a camera-shaped block is present. LOW means absent, which
                     is the normal state of most images on the internet.
  provenance         C2PA / content credentials. Almost always absent today.
  self_declaration   the file's own statement that it was generated. The only model-free
                     stream that points at AI, and it points there hard when present.
  manipulation       compression, resampling and noise evidence of post-processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from forge.image.forensics import (
    HIGH,
    LOW,
    MEDIUM,
    NOT_CHECKED,
    NOT_FOUND,
    PRESENT,
    Finding,
)

EVIDENCE_VERSION = "image_evidence_v1"

# Which way a stream points when it is strong. Only self_declaration points at AI, and that
# asymmetry is the whole reason the streams are not summed.
TOWARD_AI = "toward_ai"
TOWARD_CAPTURE = "toward_capture"
NEUTRAL = "neutral"


@dataclass(frozen=True)
class Stream:
    """One evidence stream, as the breakdown bars render it."""

    key: str
    label: str
    strength: int          # 0-100, how much this stream has to say. NOT a probability.
    direction: str
    summary: str
    available: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _find(findings: list[Finding], name: str) -> Finding | None:
    return next((f for f in findings if f.name == name), None)


def _camera_stream(findings: list[Finding]) -> Stream:
    """How much capture metadata is present, scored on completeness.

    Deliberately NOT phrased as "authenticity". A complete EXIF block means someone wrote
    a complete EXIF block. It is trivially forgeable and its absence is the normal state of
    a screenshot.
    """
    exif_finding = _find(findings, "exif")
    consistency = _find(findings, "camera_consistency")
    if not exif_finding or exif_finding.status != PRESENT:
        return Stream(
            key="camera_metadata",
            label="Camera metadata",
            strength=0,
            direction=NEUTRAL,
            summary="No EXIF block",
            note="normal for screenshots and platform downloads; not evidence either way",
        )
    fields = exif_finding.value if isinstance(exif_finding.value, dict) else {}
    # Ten interesting tags are tracked upstream; completeness is how many arrived.
    strength = min(100, int(round(len(fields) / 8 * 100)))
    consistent = consistency is not None and consistency.value == "consistent"
    return Stream(
        key="camera_metadata",
        label="Camera metadata",
        strength=strength,
        direction=TOWARD_CAPTURE if consistent else NEUTRAL,
        summary=(
            f"{len(fields)} fields"
            + (", camera-consistent" if consistent else ", partial")
        ),
        note="EXIF is a claim by whatever wrote the file, and is trivially editable",
    )


def _provenance_stream(findings: list[Finding]) -> Stream:
    c2pa = _find(findings, "c2pa")
    present = c2pa is not None and c2pa.status == PRESENT
    return Stream(
        key="provenance",
        label="Provenance (C2PA)",
        strength=70 if present else 0,
        direction=NEUTRAL,
        summary="Manifest markers present" if present else "No content credentials",
        note=(
            "detected structurally, not cryptographically verified"
            if present
            else "absence is normal; almost nothing signs its output today"
        ),
    )


def _declaration_stream(findings: list[Finding]) -> Stream:
    """The one model-free stream that can point at AI."""
    declaration = _find(findings, "ai_declaration")
    if declaration and declaration.status == HIGH:
        markers = declaration.value if isinstance(declaration.value, list) else []
        return Stream(
            key="self_declaration",
            label="Self-declared generation",
            strength=95,
            direction=TOWARD_AI,
            summary="; ".join(markers[:2]),
            note="a claim by the producing software, and a strong one",
        )
    return Stream(
        key="self_declaration",
        label="Self-declared generation",
        strength=0,
        direction=NEUTRAL,
        summary="No generation markers",
        note="absence means nothing: most generators write none and editors strip them",
    )


_STRENGTH = {LOW: 15, NOT_FOUND: 0, NOT_CHECKED: 0, PRESENT: 40, MEDIUM: 50, HIGH: 85}


def _manipulation_stream(findings: list[Finding]) -> Stream:
    parts = [
        _find(findings, name)
        for name in ("error_level", "resample", "noise_consistency", "colour")
    ]
    scored = [_STRENGTH.get(f.status, 0) for f in parts if f and f.status != NOT_CHECKED]
    strength = max(scored) if scored else 0
    hot = [f.name for f in parts if f and f.status in (MEDIUM, HIGH)]
    return Stream(
        key="manipulation",
        label="Manipulation evidence",
        strength=strength,
        direction=NEUTRAL,
        summary=", ".join(hot) if hot else "No strong post-processing signal",
        note="post-processing is orthogonal to authorship: humans edit photographs too",
    )


def _visual_stream(detector_available: bool, probability: float | None) -> Stream:
    """The detector. The only stream entitled to produce a verdict."""
    if not detector_available or probability is None:
        return Stream(
            key="visual_model",
            label="Visual model (DINOv3)",
            strength=0,
            direction=NEUTRAL,
            summary="Detector not trained",
            available=False,
            note="the only stream that can answer the question; it is not available yet",
        )
    return Stream(
        key="visual_model",
        label="Visual model (DINOv3)",
        strength=int(round(abs(probability - 0.5) * 200)),
        direction=TOWARD_AI if probability >= 0.5 else TOWARD_CAPTURE,
        summary=f"P(AI) = {probability:.2%}",
        note="calibrated on validation data; see the model card for the operating point",
    )


@dataclass(frozen=True)
class Evidence:
    streams: list[Stream]
    conflict: str
    conflict_reason: str
    version: str = EVIDENCE_VERSION

    def as_dict(self) -> dict:
        return {
            "streams": [s.as_dict() for s in self.streams],
            "conflict": self.conflict,
            "conflict_reason": self.conflict_reason,
            "version": self.version,
        }


def build_evidence(
    findings: list[Finding],
    detector_available: bool = False,
    probability: float | None = None,
) -> Evidence:
    """Assemble the evidence panel. Never returns an overall authenticity score."""
    streams = [
        _visual_stream(detector_available, probability),
        _camera_stream(findings),
        _provenance_stream(findings),
        _declaration_stream(findings),
        _manipulation_stream(findings),
    ]
    conflict, reason = _conflict(streams)
    return Evidence(streams=streams, conflict=conflict, conflict_reason=reason)


def _conflict(streams: list[Stream]) -> tuple[str, str]:
    """Do any two strong streams point in opposite directions?

    This is the output a combined score destroys. An image that declares itself generated
    while carrying a full camera EXIF block is the single most interesting case a forensic
    tool can surface, and averaging the two into one number erases it entirely.
    """
    strong = [s for s in streams if s.strength >= 60 and s.direction != NEUTRAL]
    toward_ai = [s for s in strong if s.direction == TOWARD_AI]
    toward_capture = [s for s in strong if s.direction == TOWARD_CAPTURE]

    if toward_ai and toward_capture:
        return HIGH, (
            f"{toward_ai[0].label} points to generated content while "
            f"{toward_capture[0].label} points to camera capture. One of them is wrong, "
            "and which one matters more than any average of the two."
        )
    if len(strong) <= 1:
        return LOW, (
            "not enough strong evidence for streams to disagree. Most images carry little "
            "metadata, and that is unremarkable."
        )
    return LOW, "the strong evidence streams agree on direction."
