"""Phase 2. The synthetic mirror engine.

The point is to make the human and AI sides of the training set match on topic,
genre, length and structure, so that the only systematic difference left is the
generation process. A detector trained on "write an article about X" versus real web
text mostly learns domain, not authorship, and it collapses the moment the domains
shift.

Attribute extraction must paraphrase and never copy sentences from the source. If it
copies, the mirror shares surface text with its human counterpart and the classifier
learns copy detection instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MirrorAttributes:
    topic: str
    genre: str
    register: str
    target_tokens: int
    structure: str
    difficulty: str
    key_anchors: list[str] = field(default_factory=list)


def extract_attributes(human_text: str) -> MirrorAttributes:
    raise NotImplementedError("Phase 2")


def render_prompt(attrs: MirrorAttributes, template_path: str) -> str:
    raise NotImplementedError("Phase 2")


def validate(human_text: str, generated: str, attrs: MirrorAttributes) -> tuple[bool, str]:
    """Reject on length drift, assistant preamble, or near-duplication of the source."""
    raise NotImplementedError("Phase 2")
