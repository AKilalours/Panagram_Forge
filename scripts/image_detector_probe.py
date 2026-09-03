"""Settle the image detector's label polarity from labelled images, not from its label names.

WHY THIS EXISTS. A published classifier's `id2label` is documentation, and documentation can
be wrong: several public AI-image detectors ship with their class names in the opposite order
to their actual outputs. Trusting the names produced a page that called a Canon photograph
"AI DETECTED, 99.9%" and a ChatGPT image carrying an IPTC trainedAlgorithmicMedia marker
"NO AI DETECTED, 1.5%". Both wrong, in the direction that accuses a real photographer.

The rule this project already applies to MAGE applies here: when two sources disagree about
which class is positive, MEASURE IT. Never flip the sign until the numbers look better,
because that procedure produces a confident wrong answer exactly as easily as a right one.

    python scripts/image_detector_probe.py --ai <dir of known AI images> \
                                           --human <dir of known camera photographs>

Writes reports/experiments/image_detector_polarity.json, which the detector then trusts over
the model's own label names. Until that file exists the UI marks the direction UNVERIFIED.

What counts as evidence here: a clear separation between the two groups in a consistent
direction, over enough images that a couple of hard cases cannot flip it. If the groups do
not separate, the answer is that this detector is unusable, not that the polarity is the
other way round.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".mpo"}
MIN_PER_GROUP = 5
MIN_SEPARATION = 0.20      # difference in group means below which nothing is concluded
OUT = pathlib.Path("reports/experiments/image_detector_polarity.json")


def _images(root: str) -> list[pathlib.Path]:
    return sorted(
        p for p in pathlib.Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", required=True, help="directory of images known to be generated")
    parser.add_argument("--human", required=True, help="directory of real photographs")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    from forge.image.detector import load_detector

    detector = load_detector()
    print(f"detector {detector.model_id}")
    print(f"labels   {detector.labels}, name-based AI index = {detector.ai_index}\n")

    groups: dict[str, list[float]] = {}
    for name, root in (("ai", args.ai), ("human", args.human)):
        files = _images(root)[: args.limit]
        if len(files) < MIN_PER_GROUP:
            print(f"{name}: only {len(files)} images in {root}; need at least {MIN_PER_GROUP}")
            return 2
        scores = []
        for path in files:
            try:
                scores.append(detector.probability(path.read_bytes()))
            except Exception as error:  # noqa: BLE001 - a bad file is skipped, and said
                print(f"  skipped {path.name}: {type(error).__name__}: {error}")
        groups[name] = scores
        print(f"{name:<6} n={len(scores):<3} mean={statistics.fmean(scores):.4f} "
              f"median={statistics.median(scores):.4f} "
              f"min={min(scores):.4f} max={max(scores):.4f}")

    ai_mean, human_mean = statistics.fmean(groups["ai"]), statistics.fmean(groups["human"])
    separation = abs(ai_mean - human_mean)
    print(f"\nseparation {separation:.4f} (AI mean minus human mean = {ai_mean - human_mean:+.4f})")

    if separation < MIN_SEPARATION:
        print(
            f"\nINCONCLUSIVE. The groups differ by less than {MIN_SEPARATION}. This detector "
            "does not separate these images, so no polarity is recorded and none should be "
            "assumed. That is a statement about the detector, not about the images."
        )
        return 3

    inverted = ai_mean < human_mean
    verified_index = (1 - detector.ai_index) if inverted else detector.ai_index
    print(
        f"\n{'INVERTED' if inverted else 'CONSISTENT'}: the model's label names say index "
        f"{detector.ai_index} is AI; the measurement says index {verified_index}."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model_id": detector.model_id,
        "labels": list(detector.labels),
        "name_based_ai_index": detector.ai_index,
        "verified_ai_index": verified_index,
        "inverted_relative_to_labels": inverted,
        "n": {k: len(v) for k, v in groups.items()},
        "mean_probability_as_scored": {"ai": round(ai_mean, 6), "human": round(human_mean, 6)},
        "separation": round(separation, 6),
        "method": (
            "Group means over labelled directories, scored with the name-based mapping. If "
            "the known-AI group scores LOWER than the known-human group, the mapping is "
            "inverted and the verified index is the other class. Measured, never flipped to "
            "make a result look better."
        ),
    }, indent=2) + "\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
