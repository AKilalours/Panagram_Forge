"""Mirror validation and prompt rendering.

Validation exists because a generator will cheerfully hand back things that poison the
dataset in ways that do not look like errors:

  **Assistant preamble.** "Sure! Here's a 300-word explainer on soil chemistry:" is the
  single strongest AI tell in any corpus built carelessly. A detector trained on it
  scores 99 percent and detects nothing except chat formatting. It is also trivially
  removed by anyone evading detection.

  **Length drift.** If AI documents are systematically shorter than their human
  counterparts, the detector learns length. Real accuracy, zero transfer.

  **Near-duplication of the source.** If the generator reconstructs the human document,
  the pair is not a mirror, it is a copy, and the classifier learns copy detection.

Each rejection is counted by reason. A high preamble-rejection rate means the prompt
needs work, not that the model is bad, and you only find that out if the counts exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from forge.dedup.minhash import MinHash, estimated_jaccard
from forge.generation.attributes import MirrorAttributes

# Openers a chat model reaches for. Matched at the start only, case-insensitive.
_PREAMBLE = re.compile(
    r"^\s*(?:sure|certainly|of course|absolutely|here(?:'s| is)|below is|i'd be happy|"
    r"as requested|great question|okay|ok)\b[^\n]{0,120}?[:\n]",
    re.IGNORECASE,
)
_TRAILING_META = re.compile(
    r"\n\s*(?:let me know if|i hope this helps|feel free to|would you like me to)\b.*$",
    re.IGNORECASE | re.DOTALL,
)
_MD_HEADING = re.compile(r"^\s*#{1,6}\s*\S")


@dataclass(frozen=True)
class ValidationPolicy:
    length_ratio_min: float = 0.6
    length_ratio_max: float = 1.6
    reject_assistant_preamble: bool = True
    max_minhash_jaccard_to_source: float = 0.5
    max_retries: int = 2


@dataclass
class ValidationStats:
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict:
        total = self.accepted + sum(self.rejected.values())
        return {
            "accepted": self.accepted,
            "attempts": total,
            "acceptance_rate": round(self.accepted / total, 4) if total else 0.0,
            "rejected": dict(self.rejected),
        }


_TEMPLATE_COMMENT = re.compile(r"^\s*\[[^\]]*\]\s*\n", re.MULTILINE)


def load_template(path: str | Path) -> str:
    """Load the prompt, dropping the leading [bracketed] provenance comment.

    That header records the frozen prompt version for humans reading the repo. Sending
    it to the model would put "mirror_v1 - FROZEN" inside every generation's context,
    which is both noise and a distinctive artifact the detector could learn.
    """
    return _TEMPLATE_COMMENT.sub("", Path(path).read_text(), count=1).lstrip()


def render_prompt(attrs: MirrorAttributes, template: str) -> str:
    fields = attrs.prompt_fields()
    missing = [k for k in re.findall(r"\{(\w+)\}", template) if k not in fields]
    if missing:
        raise KeyError(f"template references unknown fields: {missing}")
    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", v)
    return out


def strip_wrapper(text: str) -> str:
    """Remove trailing chat meta and a leading markdown heading.

    A leading preamble is NOT stripped. It is a rejection, because a model that produced
    one probably ignored the rest of the instruction too, and silently patching it hides
    a prompt problem behind a clean-looking dataset.
    """
    text = _TRAILING_META.sub("", text).strip()
    lines = text.split("\n")
    if lines and _MD_HEADING.match(lines[0]):
        text = "\n".join(lines[1:]).strip()
    return text


def validate(
    generated: str,
    source_text: str,
    attrs: MirrorAttributes,
    policy: ValidationPolicy = ValidationPolicy(),
    _hasher: MinHash | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason). reason is "" when ok."""
    text = generated.strip()
    if not text:
        return False, "empty"

    if policy.reject_assistant_preamble and _PREAMBLE.match(text):
        return False, "assistant_preamble"

    n = len(text.split())
    target = max(attrs.target_tokens, 1)
    ratio = n / target
    if ratio < policy.length_ratio_min:
        return False, "too_short"
    if ratio > policy.length_ratio_max:
        return False, "too_long"

    hasher = _hasher or MinHash()
    j = estimated_jaccard(hasher.signature(text), hasher.signature(source_text))
    if j > policy.max_minhash_jaccard_to_source:
        return False, "near_duplicate_of_source"

    return True, ""
