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
# BOTH must clear the bar. Mean separation alone is fooled by a bimodal detector: one
# candidate here scored a mean gap of 0.314 while placing more than half its AI images below
# 0.07, its mean carried entirely by a few confident hits. Median separation is what says
# whether the whole distribution moved.
MIN_SEPARATION = 0.20
MIN_MEDIAN_SEPARATION = 0.15
OUT = pathlib.Path("reports/experiments/image_detector_polarity.json")


def _images(root: str) -> list[pathlib.Path]:
    return sorted(
        p for p in pathlib.Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def _probe_all(args) -> int:
    """Every candidate against the same labelled set, reported together.

    One table, so the comparison is like for like. A detector is chosen by which one
    separates the probe set, never by which one produces a nicer headline number.
    """
    import os
    import statistics

    from forge.image.detector import CANDIDATES, load_detector

    files = {name: _images(root)[: args.limit]
             for name, root in (("ai", args.ai), ("human", args.human))}
    for name, paths in files.items():
        if len(paths) < MIN_PER_GROUP:
            print(f"{name}: only {len(paths)} images; need at least {MIN_PER_GROUP}")
            return 2

    print(f"probe set: {len(files['ai'])} AI, {len(files['human'])} human\n")
    print(f"{'model':<44}{'AI med':>9}{'human med':>11}{'AI mean':>9}"
          f"{'hum mean':>10}{'sep':>8}  verdict")
    print("-" * 100)

    for model_id in CANDIDATES:
        os.environ["FORGE_IMAGE_DETECTOR"] = model_id
        load_detector.cache_clear()
        try:
            detector = load_detector()
        except Exception as error:  # noqa: BLE001 - a model that will not load is reported
            print(f"{model_id:<44}{'could not load: ' + type(error).__name__:>50}")
            continue

        scores = {}
        for group, paths in files.items():
            values = []
            for path in paths:
                try:
                    values.append(detector.probability(path.read_bytes()))
                except Exception:  # noqa: BLE001
                    pass
            scores[group] = values

        ai_mean, human_mean = statistics.fmean(scores["ai"]), statistics.fmean(scores["human"])
        ai_med, human_med = statistics.median(scores["ai"]), statistics.median(scores["human"])
        separation = abs(ai_mean - human_mean)
        median_separation = abs(ai_med - human_med)
        if separation < MIN_SEPARATION or median_separation < MIN_MEDIAN_SEPARATION:
            call = ("does not separate" if separation < MIN_SEPARATION
                    else f"bimodal: mean gap {separation:.3f}, median gap {median_separation:.3f}")
        elif ai_med > human_med:
            call = f"separates, labels consistent (AI index {detector.ai_index})"
        else:
            call = f"separates, labels INVERTED (AI index {1 - detector.ai_index})"
        print(f"{model_id:<44}{ai_med:>9.4f}{human_med:>11.4f}{ai_mean:>9.4f}"
              f"{human_mean:>10.4f}{separation:>8.4f}  {call}")

    print("\nNothing is recorded by --all. Re-run with --model <the one that separates> to "
          "measure and record its polarity.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", required=True, help="directory of images known to be generated")
    parser.add_argument("--human", required=True, help="directory of real photographs")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--model", default=None, help="probe one model id instead of the default")
    parser.add_argument("--all", action="store_true", help="probe every candidate in turn")
    args = parser.parse_args()

    from forge.image.detector import CANDIDATES

    if args.all:
        return _probe_all(args)

    from forge.image.detector import load_detector

    if args.model:
        import os

        os.environ["FORGE_IMAGE_DETECTOR"] = args.model
        load_detector.cache_clear()
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
    ai_med, human_med = statistics.median(groups["ai"]), statistics.median(groups["human"])
    separation = abs(ai_mean - human_mean)
    median_separation = abs(ai_med - human_med)
    print(f"\nmean separation   {separation:.4f} (AI minus human = {ai_mean - human_mean:+.4f})")
    print(f"median separation {median_separation:.4f} "
          f"(AI minus human = {ai_med - human_med:+.4f})")

    if separation < MIN_SEPARATION or median_separation < MIN_MEDIAN_SEPARATION:
        print(
            f"\nINCONCLUSIVE. Needs mean separation >= {MIN_SEPARATION} AND median "
            f"separation >= {MIN_MEDIAN_SEPARATION}. A mean gap without a median gap means "
            "the detector is bimodal: a few confident hits carrying an average, with most "
            "images undecided. Nothing is recorded, and nothing should be assumed. That is a "
            "statement about the detector, not about the images."
        )
        return 3

    inverted = ai_med < human_med
    verified_index = (1 - detector.ai_index) if inverted else detector.ai_index
    print(
        f"\n{'INVERTED' if inverted else 'CONSISTENT'}: the model's label names say index "
        f"{detector.ai_index} is AI; the measurement says index {verified_index}."
    )

    # OPERATING POINT, fitted on the human group. This project's whole design is that being
    # wrong about a person is the expensive error, so the AI threshold is placed above every
    # human image in the probe set: zero false positives here, by construction.
    scores = groups if not inverted else {k: [1 - v for v in vs] for k, vs in groups.items()}
    human_scores, ai_scores = sorted(scores["human"]), scores["ai"]
    threshold_ai = min(1.0, max(human_scores) + 1e-6)
    human_p90 = human_scores[max(0, int(0.9 * len(human_scores)) - 1)]
    recall = sum(1 for s in ai_scores if s >= threshold_ai) / len(ai_scores)
    print(f"\noperating point fitted on the human group:")
    print(f"  AI threshold      {threshold_ai:.4f}  (above every human image in the probe set)")
    print(f"  human ceiling     {human_p90:.4f}  (90th percentile of the human group)")
    print(f"  AI recall there   {recall:.1%}  of {len(ai_scores)} AI images")
    print(f"\n  IN SAMPLE. The threshold is fitted on the same {len(human_scores)} human "
          f"images it is evaluated on, so this recall is optimistic and this false-positive "
          f"rate is zero by construction. It is an operating point, not a result.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model_id": detector.model_id,
        "labels": list(detector.labels),
        "name_based_ai_index": detector.ai_index,
        "verified_ai_index": verified_index,
        "inverted_relative_to_labels": inverted,
        "n": {k: len(v) for k, v in groups.items()},
        "mean_probability_as_scored": {"ai": round(ai_mean, 6), "human": round(human_mean, 6)},
        "median_probability_as_scored": {"ai": round(ai_med, 6), "human": round(human_med, 6)},
        "separation": round(separation, 6),
        "median_separation": round(median_separation, 6),
        "threshold_ai": round(threshold_ai, 6),
        "human_ceiling": round(human_p90, 6),
        "ai_recall_at_threshold": round(recall, 6),
        "operating_point_is_in_sample": True,
        "method": (
            "Group medians and means over labelled directories, scored with the name-based "
            "mapping. Both must separate: a mean gap without a median gap means a bimodal "
            "detector whose average is carried by a few confident hits. If the known-AI "
            "group scores LOWER, the mapping is inverted and the verified index is the other "
            "class. The threshold is placed above every human image in the probe set, so it "
            "is fitted in sample and its recall is optimistic. Measured, never adjusted to "
            "make a result look better."
        ),
    }, indent=2) + "\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
