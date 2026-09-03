"""Label polarity, and the refusals that keep a wrong verdict off the page.

The bug this file mostly exists for: assuming output index 1 is the AI class. A published
classifier may order its classes either way. Guessing wrong inverts every verdict on the
page while everything looks normal, which is exactly what happened to the MAGE benchmark in
the text track, where a wrong polarity would have turned an AUROC of 0.05 into a reported
0.95.
"""

from __future__ import annotations

import pytest

from forge.image.detector import (
    ABSTAIN_HIGH,
    ABSTAIN_LOW,
    Detection,
    LabelPolarityUnknown,
    resolve_ai_index,
)


def test_polarity_is_read_from_the_names_either_way_round():
    assert resolve_ai_index({0: "human", 1: "artificial"}) == 1
    assert resolve_ai_index({0: "artificial", 1: "human"}) == 0
    assert resolve_ai_index({"0": "Real", "1": "Fake"}) == 1
    assert resolve_ai_index({0: "AI", 1: "Not AI"}) == 0


def test_not_ai_is_read_before_ai():
    """Substring order matters: 'not ai' contains 'ai'. Longest match wins."""
    assert resolve_ai_index({0: "ai_generated", 1: "not_ai"}) == 0
    assert resolve_ai_index({0: "non-ai", 1: "ai generated"}) == 1


def test_unrecognised_labels_are_refused_rather_than_defaulted():
    """THE ONE THAT MATTERS. No silent fall back to index 1."""
    with pytest.raises(LabelPolarityUnknown) as error:
        resolve_ai_index({0: "class_0", 1: "class_1"})
    assert "class_0" in str(error.value), "the message must name the labels it saw"


def test_two_ai_classes_are_refused_because_the_positive_class_is_ambiguous():
    with pytest.raises(LabelPolarityUnknown):
        resolve_ai_index({0: "midjourney", 1: "stable diffusion"})


def test_all_human_labels_are_refused_because_nothing_means_ai():
    with pytest.raises(LabelPolarityUnknown):
        resolve_ai_index({0: "real", 1: "authentic"})


def test_the_verdict_has_three_states_and_the_middle_one_declines():
    """A detector at a low false-positive budget must be allowed to say "I do not know"."""
    def verdict(p):
        return Detection(ai_probability=p, model_id="m").verdict

    assert verdict(0.01) == "human"
    assert verdict(0.99) == "ai"
    assert verdict(0.5) == "uncertain"
    assert verdict(ABSTAIN_LOW) == "uncertain", "the band is inclusive at the low edge"
    assert verdict(ABSTAIN_HIGH) == "ai", "the band is exclusive at the high edge"
    assert verdict(ABSTAIN_LOW - 1e-9) == "human"


def test_a_detection_never_claims_to_be_calibrated():
    """Calibration needs a validation split from a corpus that does not exist yet."""
    assert Detection(ai_probability=0.9, model_id="m").calibrated is False


def test_detector_state_never_raises_when_nothing_can_load(monkeypatch):
    """The page reads this to render system state. It must not be able to take the page down."""
    import forge.image.detector as D

    monkeypatch.setattr(D, "CANDIDATES", ())
    monkeypatch.setenv("FORGE_IMAGE_DETECTOR", "")
    D.load_detector.cache_clear()
    state = D.detector_state()
    assert state["available"] is False
    assert state["reason"]
    D.load_detector.cache_clear()


def test_a_named_model_is_not_silently_replaced_by_a_fallback(monkeypatch):
    """If the report names a detector, that detector produced the score."""
    import forge.image.detector as D

    tried = []

    def _fail(model_id):
        tried.append(model_id)
        raise RuntimeError("nope")

    monkeypatch.setattr(D, "_load_one", _fail)
    monkeypatch.setenv("FORGE_IMAGE_DETECTOR", "someone/specific-model")
    D.load_detector.cache_clear()
    with pytest.raises(RuntimeError):
        D.load_detector()
    assert tried == ["someone/specific-model"], f"fell back to {tried}"
    D.load_detector.cache_clear()


def test_a_measurement_outranks_the_model_s_own_label_names(tmp_path, monkeypatch):
    """THE REGRESSION. id2label is documentation, and documentation can be wrong.

    Trusting the names produced "AI DETECTED 99.9%" on a Canon photograph and
    "NO AI DETECTED 1.5%" on a ChatGPT image carrying an IPTC trainedAlgorithmicMedia
    marker. A clean inversion, in the direction that accuses a real photographer.
    """
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({
        "model_id": "some/detector", "verified_ai_index": 0,
        "name_based_ai_index": 1, "inverted_relative_to_labels": True,
    }))
    assert D.verified_polarity("some/detector")["verified_ai_index"] == 0


def test_a_measurement_for_a_different_model_does_not_transfer(tmp_path, monkeypatch):
    """Polarity is a property of one model. Reusing another's is how a fix becomes a bug."""
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"model_id": "other/model", "verified_ai_index": 0}))
    assert D.verified_polarity("some/detector") is None


def test_no_record_means_unverified_rather_than_assumed_correct(tmp_path, monkeypatch):
    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    assert D.verified_polarity("some/detector") is None


def test_an_unreadable_record_is_treated_as_absent(tmp_path, monkeypatch):
    """A corrupt file must not be read as a confirmation."""
    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text("{not json")
    assert D.verified_polarity("some/detector") is None


def test_an_unverified_detector_says_so_in_the_evidence_panel():
    """The direction of the number is a claim until it is measured. Say which."""
    from forge.image.evidence import build_evidence

    streams = build_evidence([], detector_available=True, probability=0.99,
                             model_id="m", polarity_verified=False).streams
    visual = next(s for s in streams if s.key == "visual_model")
    assert "POLARITY UNVERIFIED" in visual.note

    verified = build_evidence([], detector_available=True, probability=0.99,
                              model_id="m", polarity_verified=True).streams
    assert "POLARITY UNVERIFIED" not in next(
        s for s in verified if s.key == "visual_model").note


def test_the_visual_stream_never_claims_calibration_it_does_not_have():
    """It said "calibrated on validation data" for a detector with no validation split."""
    from forge.image.evidence import build_evidence

    streams = build_evidence([], detector_available=True, probability=0.7,
                             model_id="m", polarity_verified=True).streams
    note = next(s for s in streams if s.key == "visual_model").note
    assert "calibrated on validation data" not in note
    assert "uncalibrated" in note


def test_the_verdict_uses_a_fitted_threshold_when_one_was_measured():
    """The default band is a placeholder. A measured operating point replaces it."""
    fitted = Detection(ai_probability=0.55, model_id="m", threshold_ai=0.42, human_ceiling=0.10)
    assert fitted.verdict == "ai", "0.55 is above a fitted threshold of 0.42"

    default = Detection(ai_probability=0.55, model_id="m")
    assert default.verdict == "uncertain", "0.55 sits inside the default band"


def test_the_middle_band_declines_rather_than_guessing():
    d = Detection(ai_probability=0.30, model_id="m", threshold_ai=0.42, human_ceiling=0.10)
    assert d.verdict == "uncertain"


def test_the_threshold_sits_above_the_human_ceiling():
    """Inverting them would make one of the two verdicts unreachable."""
    d = Detection(ai_probability=0.0, model_id="m", threshold_ai=0.42, human_ceiling=0.10)
    assert d.human_ceiling < d.threshold_ai


def test_the_measured_detector_is_the_one_that_loads(tmp_path, monkeypatch):
    """THE REGRESSION. A probe measured one model while the server served another.

    The record named umm-maybe; the loader walked its candidate list and got Organika,
    which had already FAILED the probe. The record did not match, was correctly ignored,
    and the page reported "polarity not verified" with a finished measurement on disk.
    """
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_IMAGE_DETECTOR", raising=False)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"model_id": "measured/model", "verified_ai_index": 1}))
    assert D.measured_model_id() == "measured/model"

    loaded = []
    monkeypatch.setattr(D, "_load_one", lambda m: loaded.append(m) or "detector")
    D.load_detector.cache_clear()
    D.load_detector()
    assert loaded == ["measured/model"], f"loaded {loaded} instead of the measured model"
    D.load_detector.cache_clear()


def test_an_explicit_model_still_outranks_the_measured_one(tmp_path, monkeypatch):
    """Someone naming a model is probing it. Overriding that would defeat the probe."""
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"model_id": "measured/model", "verified_ai_index": 1}))
    monkeypatch.setenv("FORGE_IMAGE_DETECTOR", "explicit/model")

    loaded = []
    monkeypatch.setattr(D, "_load_one", lambda m: loaded.append(m) or "detector")
    D.load_detector.cache_clear()
    D.load_detector()
    assert loaded == ["explicit/model"]
    D.load_detector.cache_clear()


def test_without_a_record_the_candidate_order_still_applies(tmp_path, monkeypatch):
    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_IMAGE_DETECTOR", raising=False)
    assert D.measured_model_id() is None
