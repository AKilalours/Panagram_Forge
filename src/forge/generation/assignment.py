"""Which generator writes which mirror, and with what decoding.

Two properties this file exists to guarantee.

**Held-out families never generate training data.** R3, the unseen-generator regime, is
only meaningful if gemma, deepseek and the API family were decided before results were
seen and never leaked in. `held_in_families` filters the roster and
`assert_no_held_out` is called by the runner as a hard check.

**Assignment is deterministic, seeded by the human document id.** Re-running generation
for a document assigns the same family and the same decoding, so a partially completed
run can be resumed without producing a different dataset. Random assignment at runtime
would make dataset v0.1 unreproducible even with the same config.

Decoding is sampled from a grid rather than fixed, so the detector cannot latch onto one
temperature's artifacts and call it "AI".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from forge.generation.generators.base import Decoding


class HeldOutLeakError(RuntimeError):
    pass


@dataclass(frozen=True)
class FamilySpec:
    family: str
    provider: str
    model_id: str
    revision: str
    released: str
    role: str
    output_redistributable: bool = False


def parse_roster(cfg: dict) -> list[FamilySpec]:
    return [
        FamilySpec(
            family=f["family"],
            provider=f["provider"],
            model_id=f["model_id"],
            revision=str(f.get("revision", "")),
            released=str(f.get("released", "")),
            role=f["role"],
            output_redistributable=bool(f.get("output_redistributable", False)),
        )
        for f in cfg.get("families", [])
    ]


def held_in_families(roster: list[FamilySpec]) -> list[FamilySpec]:
    return [f for f in roster if f.role == "held_in"]


def held_out_families(roster: list[FamilySpec]) -> list[FamilySpec]:
    return [f for f in roster if f.role == "held_out"]


def assert_no_held_out(roster: list[FamilySpec], used: set[str]) -> None:
    leaked = used & {f.family for f in held_out_families(roster)}
    if leaked:
        raise HeldOutLeakError(
            f"held-out families {sorted(leaked)} generated training data. "
            "This invalidates the unseen-generator regime (R3)."
        )


def _digest(doc_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{doc_id}|{salt}".encode()).hexdigest()[:12], 16)


def assign_family(doc_id: str, families: list[FamilySpec]) -> FamilySpec:
    if not families:
        raise ValueError("no held-in families available to generate mirrors")
    return families[_digest(doc_id, "family") % len(families)]


def assign_decoding(doc_id: str, grid: dict) -> Decoding:
    temps = list(grid.get("temperature", [0.7]))
    tops = list(grid.get("top_p", [0.9]))
    max_new = int(grid.get("max_new_tokens", 1024))
    options: list[tuple[float, float]] = [(t, p) for t in temps for p in tops]
    if grid.get("include_greedy"):
        options.append((0.0, 1.0))
    t, p = options[_digest(doc_id, "decoding") % len(options)]
    return Decoding(temperature=t, top_p=p, max_new_tokens=max_new, seed=_digest(doc_id, "seed") % (2**31))
