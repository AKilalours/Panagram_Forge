"""Generate a local fixture corpus for exercising the Phase 1 pipeline offline.

This is NOT training data and must never be treated as such. Its only job is to make
every branch of the cleaning pipeline fire so the pipeline can be verified without
network access: HTML junk, PII, exact duplicates, near duplicates, documents that are
too short, boilerplate, and non-English text.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SUBJECTS = [
    "coastal erosion", "municipal budgeting", "beekeeping", "railway signalling",
    "medieval bookbinding", "soil chemistry", "harbour dredging", "choral notation",
    "textile dyeing", "glacier monitoring", "orchard grafting", "lighthouse keeping",
]
VERBS = ["examines", "documents", "reviews", "traces", "surveys", "compares"]
CLAUSES = [
    "the practice changed considerably over the following decades",
    "local records from the period remain incomplete in several respects",
    "the committee published its findings the following spring",
    "later authors disputed both the method and the conclusion",
    "the technique spread slowly across neighbouring districts",
    "funding was withdrawn before the second phase could begin",
    "surviving correspondence suggests the decision was contested",
    "the equipment required regular maintenance through the winter months",
    "measurements were taken twice daily at fixed observation points",
    "an appendix listed the materials and their approximate costs",
]


def paragraph(rng: random.Random, subject: str, n: int = 8) -> str:
    out = [f"This account {rng.choice(VERBS)} the history of {subject}."]
    for _ in range(n):
        out.append(rng.choice(CLAUSES).capitalize() + ".")
    return " ".join(out)


def document(rng: random.Random, subject: str, paras: int = 4) -> str:
    return "\n\n".join(paragraph(rng, subject) for _ in range(paras))


def build(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    docs: list[dict] = []
    for i in range(n):
        subject = SUBJECTS[i % len(SUBJECTS)]
        docs.append({"id": f"clean{i}", "text": document(rng, subject), "domain": "web"})

    # deliberate pipeline probes, each labelled by what it should trigger
    base = docs[0]["text"]
    docs.append({"id": "exact_dup", "text": base})                      # exact_duplicate
    docs.append({"id": "near_dup", "text": base + " One extra closing sentence."})  # near_duplicate
    docs.append({"id": "short", "text": "Too short."})                  # too_short_chars
    docs.append({"id": "html", "text": f"<html><body><script>var x=1;</script><p>{document(rng,'tidal charts')}</p></body></html>"})
    docs.append({"id": "pii", "text": document(rng, "parish archives") + " Contact jane.doe@example.com or 555-123-4567 for details."})
    docs.append({"id": "symbols", "text": "### >>> ||| ### >>> ||| " * 60})   # high_symbol_ratio
    docs.append({"id": "repetitive", "text": ("Click here to continue.\n" * 80)})  # repetitive_lines
    docs.append({"id": "nonenglish", "text": ("Der Bericht beschreibt die Entwicklung der lokalen Verwaltung waehrend "
                                              "dieser Periode und nennt mehrere beteiligte Personen sowie deren Aufgaben. ") * 8})
    docs.append({"id": "zerowidth", "text": document(rng, "canal locks").replace(" ", "​ ", 5)})
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/raw/fixture/corpus.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for d in build(args.n):
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {args.n + 9} fixture documents to {out}")


if __name__ == "__main__":
    main()
