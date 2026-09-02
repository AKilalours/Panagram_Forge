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


# Subword tokens per whitespace word, English. Both arms measure their length targets in
# WORDS (forge.cleaning.filters.approx_token_count is a word count, and the mirror
# extractor's target_tokens is len(text.split())), while max_new_tokens is in TOKENS. The
# conversion has to happen somewhere; here, once, named.
TOKENS_PER_WORD = 1.35

# Room above the target so a model can finish its last sentence instead of being cut off
# mid-clause. Truncation is not a neutral failure: text that stops mid-word is trivially
# detectable, and a detector that learns THAT is worse than one that learns length. 1.3 is
# generous enough to end cleanly and still far below the validator's 1.6 ratio ceiling.
LENGTH_HEADROOM = 1.3

# Floor, so a very short source still gets a usable budget rather than a few tokens.
MIN_NEW_TOKENS = 96


def token_budget(target_words: int, grid_max: int) -> int:
    """Per-document max_new_tokens from a per-document word target.

    THE BUG THIS FIXES, and it is the most expensive one in the project so far.

    Every document was generated with ONE max_new_tokens, 640, taken from the decoding
    grid. Both arms already computed a correct per-document length target and put it in
    the prompt: the mirror arm from its source document, the control arm sampled from the
    human corpus's own length distribution. The models ignored it and wrote to the cap.

    Measured on the finished corpora:

        human    median 250 words
        mirror   median 364 words   AUROC from length alone 0.774
        random   median 394 words   AUROC from length alone 0.841

    A detector scores 0.84 on the control arm WITHOUT READING A WORD. Both arms would have
    trained a length detector and the experiment would have compared two of them.

    The cap was not a ceiling the models bumped into. It was a target they ran to. Nothing
    reported an error, the prompts looked right, and the stated target was simply advice.
    """
    if target_words <= 0:
        raise ValueError(f"target_words must be positive, got {target_words}")
    wanted = int(target_words * TOKENS_PER_WORD * LENGTH_HEADROOM)
    return max(MIN_NEW_TOKENS, min(wanted, grid_max))


def assign_decoding(doc_id: str, grid: dict, target_words: int | None = None) -> Decoding:
    """Pick decoding for one document.

    `target_words` is optional so the function keeps working for callers that have no
    length target, but every production caller passes one. Temperature, top_p and seed are
    chosen exactly as before, from the same hashes, so adding a length budget does NOT
    change which decoding setting a document receives.
    """
    temps = list(grid.get("temperature", [0.7]))
    tops = list(grid.get("top_p", [0.9]))
    max_new = int(grid.get("max_new_tokens", 1024))
    if target_words is not None:
        max_new = token_budget(target_words, max_new)
    options: list[tuple[float, float]] = [(t, p) for t in temps for p in tops]
    if grid.get("include_greedy"):
        options.append((0.0, 1.0))
    t, p = options[_digest(doc_id, "decoding") % len(options)]
    return Decoding(temperature=t, top_p=p, max_new_tokens=max_new, seed=_digest(doc_id, "seed") % (2**31))
