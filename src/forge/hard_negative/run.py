"""Phase 4 orchestration: one turn of the flywheel.

    scan reserve -> build atlas -> select across clusters -> targeted mirrors
      -> hard negative dataset -> (retrain, next phase)

What "targeted" means here, concretely. A cluster is a failure mode, for example
"formal register, government source, long paragraphs, 32 percent of all false
positives". Targeted generation takes the mined HUMAN documents in that cluster and
mirrors them through the Phase 2 engine. The training set then contains matched pairs
drawn from exactly the region where the model was confidently wrong: the human document
it misjudged, and an AI document that looks like it.

That pairing is the mechanism. Adding the mined humans alone would teach the model
"formal government prose is human", which fixes the FPR by breaking the FNR on formal
government AI text. The mirror is what makes it learn the authorship difference rather
than the register.

One invariant enforced below: mined humans and their targeted mirrors go to the TRAIN
split. They came from the reserve pool, which is outside train/val/test, so they carry no
split of their own, and the whole point is that they enter training. Held-out clusters
are written separately and never enter training at all: they are the measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.common.schemas import Split
from forge.failure_atlas.atlas import Atlas, build_atlas
from forge.failure_atlas.embedding import Embedder
from forge.generation.run import HumanRef
from forge.hard_negative.mining import MinedLedger, MiningStats, ReserveDoc, Scorer, scan
from forge.hard_negative.selection import Selection, SelectionPolicy, select_proportional


@dataclass
class MiningRound:
    round_name: str
    stats: MiningStats
    atlas: Atlas
    selection: Selection
    train_refs: list[HumanRef] = field(default_factory=list)
    holdout_refs: list[HumanRef] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "round": self.round_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mining": self.stats.as_dict(),
            "atlas": self.atlas.as_dict(),
            "selection": self.selection.as_dict(),
            "n_train_refs": len(self.train_refs),
            "n_holdout_refs": len(self.holdout_refs),
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return p


def run_round(
    reserve: list[ReserveDoc],
    scorer: Scorer,
    operating_threshold: float,
    round_name: str = "mining_run_001",
    min_confidence: float = 0.90,
    policy: SelectionPolicy | None = None,
    embedder: Embedder | None = None,
    min_cluster_size: int = 25,
    k: int | None = None,
    ledger: MinedLedger | None = None,
) -> MiningRound:
    policy = policy or SelectionPolicy()
    failures, stats = scan(
        reserve, scorer, operating_threshold, min_confidence,
        ledger=ledger, round_name=round_name,
    )
    if not failures:
        raise RuntimeError(
            "mining found no high-confidence false positives. That is either very good "
            "news or a broken scorer; check the error rate in the stats before assuming "
            "the former."
        )

    by_id = {d.doc_id: d for d in reserve}
    texts = [by_id[f.sample_id].text for f in failures]
    atlas = build_atlas(failures, texts, embedder=embedder, min_cluster_size=min_cluster_size, k=k)
    sel = select_proportional(failures, atlas.labels, policy)

    def _refs(records) -> list[HumanRef]:
        out = []
        for r in records:
            d = by_id[r.sample_id]
            # Mined humans enter TRAIN. They came from the reserve, outside the splits.
            out.append(HumanRef(d.doc_id, d.source_group_id, d.text, d.domain, Split.TRAIN))
        return out

    round_ = MiningRound(round_name, stats, atlas, sel, _refs(sel.train), _refs(sel.holdout))
    if ledger is not None:
        ledger.add(r.sample_id for r in sel.train)
        ledger.save(round_name)
    return round_
