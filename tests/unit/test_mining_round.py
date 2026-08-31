"""One full turn of the flywheel, end to end, against a scorer with known failures."""

import json

import pytest

from forge.common.schemas import Split
from forge.hard_negative.mining import MinedLedger, ReserveDoc
from forge.hard_negative.run import run_round
from forge.hard_negative.selection import SelectionPolicy

REGISTERS = {
    "academic": "Citation-heavy academic prose with long paragraphs and a formal register, item {i}, on methodology.",
    "legal": "SECTION {i}. PURSUANT TO THE FOREGOING PROVISIONS THE COMMITTEE SHALL DETERMINE THE THRESHOLD.",
    "casual": "hey so basically the thing is you gotta just try it out and see what happens, number {i}, no big deal",
}


def _reserve(n=80):
    return [
        ReserveDoc(f"{reg}_{i}", f"grp_{reg}_{i}", tpl.format(i=i), "fineweb", "web", reg)
        for reg, tpl in REGISTERS.items()
        for i in range(n)
    ]


class TwoModeScorer:
    """Confidently wrong on two of the three registers."""

    model_version = "forge-0.2-test"

    def score(self, texts):
        return [0.96 if ("Citation-heavy" in t or "PURSUANT" in t) else 0.02 for t in texts]


def test_a_round_produces_train_and_holdout_from_different_clusters():
    r = run_round(_reserve(), TwoModeScorer(), operating_threshold=0.5, k=2,
                  policy=SelectionPolicy(max_selected=100))
    assert r.train_refs and r.holdout_refs
    assert not (set(r.selection.train_clusters) & set(r.selection.holdout_clusters))


def test_mined_documents_enter_the_train_split():
    """They come from the reserve pool, which sits outside train/val/test, and the whole
    point of mining is that they enter training."""
    r = run_round(_reserve(), TwoModeScorer(), 0.5, k=2, policy=SelectionPolicy(max_selected=100))
    assert all(ref.split is Split.TRAIN for ref in r.train_refs)


def test_holdout_documents_never_appear_in_training():
    r = run_round(_reserve(), TwoModeScorer(), 0.5, k=2, policy=SelectionPolicy(max_selected=100))
    assert not ({x.doc_id for x in r.train_refs} & {x.doc_id for x in r.holdout_refs})


def test_the_round_only_mines_the_registers_the_model_actually_failed_on():
    r = run_round(_reserve(), TwoModeScorer(), 0.5, k=2, policy=SelectionPolicy(max_selected=1000))
    mined = {ref.doc_id.split("_")[0] for ref in r.train_refs + r.holdout_refs}
    assert mined <= {"academic", "legal"}
    assert "casual" not in mined


def test_a_clean_model_raises_rather_than_reporting_an_empty_success():
    class PerfectScorer:
        model_version = "m"

        def score(self, texts):
            return [0.01] * len(texts)

    with pytest.raises(RuntimeError, match="no high-confidence false positives"):
        run_round(_reserve(), PerfectScorer(), 0.5, k=2)


def test_round_two_does_not_re_mine_round_one(tmp_path):
    ledger = MinedLedger(tmp_path / "mined.json")
    r1 = run_round(_reserve(), TwoModeScorer(), 0.5, k=2,
                   policy=SelectionPolicy(max_selected=1000), ledger=ledger, round_name="r1")
    assert r1.train_refs
    r2_ids = None
    try:
        r2 = run_round(_reserve(), TwoModeScorer(), 0.5, k=2,
                       policy=SelectionPolicy(max_selected=1000), ledger=ledger, round_name="r2")
        r2_ids = {x.doc_id for x in r2.train_refs}
    except RuntimeError:
        r2_ids = set()
    assert not (r2_ids & {x.doc_id for x in r1.train_refs})


def test_report_is_a_committable_artifact(tmp_path):
    r = run_round(_reserve(), TwoModeScorer(), 0.5, k=2, policy=SelectionPolicy(max_selected=100))
    d = json.loads(r.write(tmp_path / "round.json").read_text())
    assert d["mining"]["scanned"] == 240
    assert d["atlas"]["n_failures"] == 160
    assert d["atlas"]["clustering"]["method"] in ("kmeans", "hdbscan")
    assert d["selection"]["holdout_clusters"]
