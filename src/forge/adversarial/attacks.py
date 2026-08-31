"""Phase 6. Controlled adversarial transformations.

Ours are controlled so we know exactly what changed and can attribute a degradation to
a specific transformation. RAID's 12 attacks are the independent external check,
because attacks we wrote ourselves are attacks we implicitly designed around.
"""

from __future__ import annotations

ATTACKS = (
    "paraphrase_llm",
    "sentence_reorder",
    "synonym_swap",
    "whitespace_perturb",
    "homoglyph_substitute",
    "zero_width_insert",
    "grammar_edit",
    "humanizer_tool",
    "ai_assisted_edit",
)


def apply(text: str, attack: str, **params) -> str:
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}")
    raise NotImplementedError("Phase 6")
