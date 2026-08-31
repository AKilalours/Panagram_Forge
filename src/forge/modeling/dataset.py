"""Windowed training examples.

A document longer than the encoder's window becomes several examples. The document-level
label is copied to every window, which is a deliberate simplification with a known cost:
a mostly-human document with one AI paragraph produces mostly-human windows carrying an
"ai" document label. That is why `windows_from_spans` recomputes each window's own label
from its token spans rather than inheriting blindly. Only genuinely single-label
documents (pure human, pure mirror) inherit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.common.schemas import Label, Split, TokenLabel
from forge.modeling.alignment import IGNORE_INDEX, LABEL_TO_ID, align_spans_to_tokens
from forge.modeling.windowing import windows


@dataclass
class Example:
    doc_id: str
    source_group_id: str
    split: Split
    window_index: int
    char_start: int
    char_end: int
    document_label: int          # 0 human, 1 ai
    token_label_ids: list[int]
    ai_char_fraction: float
    meta: dict = field(default_factory=dict)


def windows_from_spans(
    doc_id: str,
    source_group_id: str,
    split: Split,
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, TokenLabel]],
    size: int = 512,
    stride: int = 384,
    ai_window_threshold: float = 0.5,
    meta: dict | None = None,
) -> list[Example]:
    token_labels = align_spans_to_tokens(offsets, spans)
    out: list[Example] = []
    for i, (a, b) in enumerate(windows(len(offsets), size=size, stride=stride)):
        win_labels = token_labels[a:b]
        real = [x for x in win_labels if x != IGNORE_INDEX]
        human_id = LABEL_TO_ID[TokenLabel.HUMAN]
        frac = (sum(1 for x in real if x != human_id) / len(real)) if real else 0.0
        char_start = next((s for s, e in offsets[a:b] if e > s), 0)
        char_end = next((e for s, e in reversed(offsets[a:b]) if e > s), char_start)
        out.append(
            Example(
                doc_id=doc_id,
                source_group_id=source_group_id,
                split=split,
                window_index=i,
                char_start=char_start,
                char_end=char_end,
                # Each window is labelled by its OWN content, not the document's.
                document_label=int(frac >= ai_window_threshold),
                token_label_ids=win_labels,
                ai_char_fraction=frac,
                meta=dict(meta or {}, window=f"{a}:{b}"),
            )
        )
    return out


def pure_document_spans(text: str, label: Label) -> list[tuple[int, int, TokenLabel]]:
    """A single span covering a whole human or mirror document."""
    tl = TokenLabel.HUMAN if label is Label.HUMAN else TokenLabel.AI_GENERATED
    return [(0, len(text), tl)]
